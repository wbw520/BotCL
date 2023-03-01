from configs import parser
import os
import torch
from loaders.get_loader import loader_generation
from model.retrieval.model_main import MainModel
from model.ConceptShape import ConceptShap


def main():
    train_loader1, train_loader2, val_loader = loader_generation(args)
    print("data prepared")
    ConceptShap(args, model_, train_loader1, val_loader).learn_concepts()


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    model_ = MainModel(args)
    args.device = "cuda:1"
    device = torch.device(args.device)
    model_.to(device)
    args.output_dir = "../saved_model"
    checkpoint = torch.load(os.path.join(args.output_dir,
                                                 f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt_no_slot.pt"),
                                    map_location=device)
    model_.load_state_dict(checkpoint, strict=True)
    main()



