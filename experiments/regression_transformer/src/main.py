################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC and other
## FLASK Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
#from torch.distributed import get_world_size
import os
import arguments, data, modeling, tokenization, training
from transformers import DataCollatorForPermutationLanguageModeling
from evaluate import eval_language_model


def main():
    args = arguments.parse()
    # Set global batch size
    world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    #world_size = max(1, get_world_size())
    args.batch_size = args.per_device_batch_size * max(1, world_size)

    tokenizer = tokenization.setup_tokenizer(args)

    # Setup training algorithm
    train_config = training.read_training_config(args)
    collator, alt_collator = training.setup_collators_for_training(
        train_config, tokenizer)
    property_tokens = getattr(collator, 'property_tokens', [])

    # Setup model
    model = modeling.setup_model(args, tokenizer)

    # Check if only evaluating
    if args.eval_only:
        eval_language_model(model, tokenizer, args)
        return

    # Setup data loader(s)
    train_loader_prop, eval_loader_prop = data.setup_loader(
        args, tokenizer, collator, collator)
    if alt_collator is not None:
        train_loader_cg, eval_loader_cg = data.setup_loader(
            args, tokenizer, alt_collator, alt_collator)
    else:
        train_loader_cg = eval_loader_cg = None

    # Initialize Trainer
    trainer = training.setup_trainer(
        model=model,
        args=args,
        property_tokens=property_tokens,
        eval_loader_cg=eval_loader_cg,
        tokenizer=tokenizer,
        train_config=train_config,
        prediction_loss_only=False,
    )

    print(vars(trainer))

    if train_loader_prop is not None:
        trainer.train(prop_dataloader=train_loader_prop,
                      cg_dataloader=train_loader_cg,
                      eval_dataloader=eval_loader_prop)

    trainer.evaluate(eval_dataloader=eval_loader_prop)


if __name__ == '__main__':
    main()
