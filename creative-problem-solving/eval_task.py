import torch
import argparse
from transformers import CLIPProcessor, CLIPModel
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import random
from dataset_cfg import ground_truth, dataset_root, image_paths, hf_model_name
from dataset_cfg import augmented_prompts_obj, augmented_prompts_task, augmented_prompts_task_obj
from plotter import plot_results
from tqdm import tqdm
import numpy as np


def get_model(model_name):
    if "vilt" in model_name:
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_name)
    else:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

    return model, processor


def run_vilt_eval(model, processor, text, images, names):
    results = {}
    for i, img in enumerate(images):
        inputs = processor(img, text, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # this is the image-text similarity score
        idx = logits.argmax(-1).item()
        predicted_answer = model.config.id2label[idx]
        if "yes" in predicted_answer.lower():
            results[names[i]] = logits.max(-1).values.item()

    # Pick the key from results that has highest value
    if not results:
        predicted_object = "None"
    else:
        predicted_object = max(results, key=results.get)
    return predicted_object


def run_clip_eval(model, processor, text, images, names):
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=0)
    idx = probs.argmax(dim=0)
    return names[idx]


def main(model_name, args):
    # Seed for reproducibility
    random.seed(args.seed)
    def create_random_three_objects(image_paths, ground_truth, exclude=""):
        objects = [k for k in image_paths.keys() if k != ground_truth and k != exclude]
        random.shuffle(objects)
        return [ground_truth] + objects[:3]

    def get_accuracy(text, predicted_object, ground_truth):
        for obj in ground_truth.keys():
            if obj in text:
                return 1 if ground_truth[obj] == predicted_object else 0
        return 0

    mode = args.task_type
    image_full_paths = {k: dataset_root + "/" + v for k, v in image_paths.items()}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = get_model(model_name)
    model.eval()
    model.to(device)

    accuracy = 0
    accuracy_by_class = {}
    N_range = 10  # Number of samples per task
    N_tasks = 5  # Number of tasks
    N_samples = N_range * N_tasks  # Total number of samples

    for _ in tqdm(range(N_range)):
        dataset_mapping = {
            "nominal": {
                "can this object be used as a scoop?": create_random_three_objects(image_paths, "spoon"),
                "can this object be used as a hammer?": create_random_three_objects(image_paths, "hammer"),
                "can this object be used as a spatula?": create_random_three_objects(image_paths, "spatula"),
                "can this object be used as a toothpick?": create_random_three_objects(image_paths, "toothpick"),
                "can this object be used as pliers?": create_random_three_objects(image_paths, "pliers")
            },
            "creative": {
                "can this object be used as a scoop?": create_random_three_objects(image_paths, "bowl", exclude="spoon"),
                "can this object be used as a hammer?": create_random_three_objects(image_paths, "saucepan", exclude="hammer"),
                "can this object be used as a spatula?": create_random_three_objects(image_paths, "knife", exclude="spatula"),
                "can this object be used as a toothpick?": create_random_three_objects(image_paths, "safety pin", exclude="toothpick"),
                "can this object be used as pliers?": create_random_three_objects(image_paths, "scissors", exclude="pliers")
            }
        }

        text_list = {
            "nominal": [t for t in dataset_mapping["nominal"]],
            "creative": [t for t in dataset_mapping["creative"]]
        }

        # Create an augmented version of the creative task
        # We want to ensure that the same test objects are used for "creative" and other prompts
        # Otherwise it will not be a fair comparison
        if mode == "creative-obj":
            dataset_mapping["creative-obj"] = {
                k: v for k, v in zip(augmented_prompts_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-obj"] = [t for t in dataset_mapping["creative-obj"]]
        elif mode == "creative-task":
            dataset_mapping["creative-task"] = {
                k: v for k, v in zip(augmented_prompts_task, dataset_mapping["creative"].values())
            }
            text_list["creative-task"] = [t for t in dataset_mapping["creative-task"]]
        elif mode == "creative-task-obj":
            dataset_mapping["creative-task-obj"] = {
                k: v for k, v in zip(augmented_prompts_task_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-task-obj"] = [t for t in dataset_mapping["creative-task-obj"]]

        assert len(text_list["nominal"]) == N_tasks

        for text in text_list[mode]:
            images = []
            names = []
            for name, path in image_full_paths.items():
                if name in dataset_mapping[mode][text]:
                    images.append(Image.open(path))
                    names.append(name)

            if "vilt" in model_name:
                predicted_object = run_vilt_eval(model, processor, text, images, names)
            else:
                predicted_object = run_clip_eval(model, processor, text, images, names)
            if args.verbose:
                print(f"Mode: {mode}, Text: {text}, Object: {predicted_object}, All objects: {names}")
            accuracy += get_accuracy(text, predicted_object, ground_truth[mode])
            if text in accuracy_by_class:
                accuracy_by_class[text] += get_accuracy(text, predicted_object, ground_truth[mode])
            else:
                accuracy_by_class[text] = 1

    if args.verbose:
        for k, v in accuracy_by_class.items():
            print(f"Accuracy for {k}: {v * 100/N_range}%")
        
    # For visualization
    accuracy_by_class = {k: v / N_range for k, v in accuracy_by_class.items()}
    overall = np.mean(list(accuracy_by_class.values()))
    accuracy_by_class["overall"] = overall
    return accuracy_by_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Computational Creativity inspired prompting for creative problem solving',
        )
    parser.add_argument(
        "--task-type", type=str, required=True, help="Choose which prompt type to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Choose seed for experiment"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print results in console"
    )
    args = parser.parse_args()
    assert args.task_type in [
        "creative",
        "nominal",
        "creative-obj",
        "creative-task",
        "creative-task-obj"
    ]

    plotting_data = {}
    for name in hf_model_name.keys():
        print(f"Model: {name}")
        acc_by_class = main(hf_model_name[name], args)
        plotting_data[name] = acc_by_class

    print("Saving visualization...")
    plot_results(args.task_type, plotting_data)
