import os
import transformers
from transformers import AutoConfig, AutoModel
from typing import List, Dict, Any
from torch import Tensor


OUTPUT_DIR = "outputs/test_output"
pawa_model_name = "sartifyllc/Pawa-Base-gemma-2-2b-it"

pretrained_model_name = "unsloth/gemma-2-2b"
finetuned_model_name = "unsloth/gemma-2-2b-it"


os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_instrctions_residual(
    state_dict_pt: Dict[str, Tensor], state_dict_ft: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    """
    calculate the residuals betweent the instructions of the pretrained and finetuned models

    \delta_phi = \phi_{finetuned} - \phi_{pretrained}
    """

    assert set(state_dict_pt.keys()) == set(state_dict_ft.keys()), (
        "The keys of the two state_dcit must match."
    )
    residuals = {}
    for key in state_dict_pt.keys():
        residuals[key] = state_dict_ft[key] - state_dict_pt[key]

    return residuals


def create_merged_state_dict(
    state_dict: Dict[str, Tensor],
    residuals: Dict[str, Tensor],
    gemma: float = 1.0,
) -> Dict[str, Tensor]:
    """
    create a merged state dict by adding the residuals to the pretrained model state dict
    """
    assert set(state_dict.keys()) == set(residuals.keys()), (
        "The keys of the state_dict and residuals must match."
    )

    merged_state_dict = {}
    for key in state_dict.keys():
        merged_state_dict[key] = state_dict[key] + gemma * residuals[key]
    return merged_state_dict


def create_load_and_save_model(
    model_name: str,
    state_dict: Dict[str, Tensor],
    output_dir: str,
):
    """
    Load a model, update its state_dict, and save it to the output directory.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(config)
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    # Load the pretrained and finetuned models
    state_dict_pt = AutoModel.from_pretrained(pretrained_model_name).state_dict()
    state_dict_ft = AutoModel.from_pretrained(finetuned_model_name).state_dict()

    # Calculate the residuals
    residuals = calculate_instrctions_residual(state_dict_pt, state_dict_ft)

    # Create the merged state dict
    merged_state_dict = create_merged_state_dict(state_dict_pt, residuals)

    # Save the merged model
    create_load_and_save_model(pawa_model_name, merged_state_dict, OUTPUT_DIR)
