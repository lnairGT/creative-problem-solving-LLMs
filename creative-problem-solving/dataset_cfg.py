# Dataset configuration
dataset_root = "artificial-dataset"

image_paths={
    "bottle": "bottle.jpeg",
    "corkscrew": "corkscrew.jpeg",
    "knife": "knife.jpeg",
    "screw": "screw.jpeg",
    "spoon": "spoon.jpeg",
    "hammer": "Claw-hammer.jpg",
    "frying pan": "frying_pan.jpeg",
    "lemon": "lemon.jpeg",
    "spatula": "spatula.jpeg",
    "street sign": "street_sign.jpeg",
    "bowl": "bowl.jpeg",
    "saucepan": "saucepan.jpeg",
    "toothpick": "toothpick.jpeg",
    "safety pin": "safety_pin.jpeg",
    "scissors": "scissors.jpeg",
    "pliers": "pliers.jpeg"
}

# Task to tool ground truth definitions
ground_truth = {
    "nominal": {
        "scoop": "spoon",
        "hammer": "hammer",
        "spatula": "spatula",
        "toothpick": "toothpick",
        "pliers": "pliers"
    },
    "creative": {
        "scoop": "bowl",
        "hammer": "saucepan",
        "spatula": "knife",
        "toothpick": "safety pin",
        "pliers": "scissors"
    },
    "creative-obj": {
        "scoop": "bowl",
        "hammer": "saucepan",
        "spatula": "knife",
        "toothpick": "safety pin",
        "pliers": "scissors"
    },
    "creative-task": {
        "scoop": "bowl",
        "hammer": "saucepan",
        "spatula": "knife",
        "toothpick": "safety pin",
        "pliers": "scissors"
    },
    "creative-task-obj": {
        "scoop": "bowl",
        "hammer": "saucepan",
        "spatula": "knife",
        "toothpick": "safety pin",
        "pliers": "scissors"
    }
}

# Models
hf_model_name = {
    "CLIP-B-32": "openai/clip-vit-base-patch32",
    "CLIP-B-16": "openai/clip-vit-base-patch16",
    "CLIP-L-14": "openai/clip-vit-large-patch14",
    "CLIP-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "VILT-B-32": "dandelin/vilt-b32-finetuned-vqa"
}

# Augmented prompts
augmented_prompts_obj = [
    "scoops must be concave and hollow. can this object be used as a scoop?",
    "hammers must be heavy and have a handle attached to a cylinder at the end. can this object be used as a hammer?",
    "spatulas must have a handle attached to a flat surface at the end. can this object be used as a spatula?",
    "toothpicks must have a pointed tip. can this object be used as a toothpick?",
    "pliers must have two-prongs. can this object be used as pliers?"
]

augmented_prompts_task = [
    "scoops can transfer beans from one jar to another jar. can this object be used as a scoop?",
    "hammers can hit a nail into the wall. can this object be used as a hammer?",
    "spatulas can spread butter onto a pan. can this object be used as a spatula?",
    "toothpicks can pick food caught between the teeth. can this object be used as a toothpick?",
    "pliers can grab a coin. can this object be used as pliers?"
]

augmented_prompts_task_obj = [
    "scoops can transfer beans from one jar to another jar. scoops are concave and hollow. can this object be used as a scoop?",
    "hammers can hit a nail into the wall. hammers have a handle attached to a cylinder at the end. can this object be used as a hammer?",
    "spatulas can spread butter onto a pan. spatulas have a handle attached to a flat surface at the end. can this object be used as a spatula?",
    "toothpicks can pick food caught between the teeth. toothpicks have a pointed tip. can this object be used as a toothpick?",
    "pliers can grab a coin. pliers have two-prongs. can this object be used as pliers?"
]