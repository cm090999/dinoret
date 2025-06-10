import torch

from model_utils.block_exansion import get_expanded_block_positions
from util.pos_embed import interpolate_pos_embed
import model_utils.models_vit as models_vit
from timm.models.layers import trunc_normal_
import util.lr_decay as lrd

def generate_model(args, device):

    modelLoader = LoadModel(args, device)
    model, model_without_ddp, param_groups, n_parameters = modelLoader.load_model()

    return model, model_without_ddp, param_groups, n_parameters

class LoadModel():
    def __init__(self, args, device) -> None:
        self.device = device
        self.args = args

        self.model = None
        self.model_without_ddp = None

        return
    
    def load_model(self):

        self.model, self.model_without_ddp, n_parameters = self.load_backbone_()
        self.fix_backbone_()
        param_groups = self.get_param_groups_()

        n_parameters_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("Model = %s" % str(self.model_without_ddp))
        print('number of params (M): %.6f' % (n_parameters / 1.e6))
        print('number of params to train (M): %.6f' % (n_parameters_train / 1.e6))

        return self.model, self.model_without_ddp, param_groups, n_parameters
    
    def load_backbone_(self):

        if "dinov2" in self.args.model:

            # TODO: Bug, that loading weights from checkpoint does not work (only works from dinov2)
            if self.args.pretrained_checkpoint != '':
                model_backbone = torch.hub.load('facebookresearch/dinov2', self.args.model, img_size=self.args.input_size, pretrained=False)
            else:
                model_backbone = torch.hub.load('facebookresearch/dinov2', self.args.model)
            model = models_vit.dinov2_vit_14(
                num_classes = self.args.nb_classes,
                backbone = model_backbone,
                model = self.args.model,
                forward_patches = self.args.forward_patches,
                drop_path_rate=self.args.drop_path,
                n_classification_heads = self.args.n_classification_heads,
                block_expansion_positions = self.args.block_expansion_positions,
                block_expansion_path_dropout = self.args.block_expansion_path_dropout,
                lora_adaptation = self.args.lora_adaptation,
                lora_adaptation_target_blocks = self.args.lora_adaptation_target_blocks,
                lora_adaptation_rank = self.args.lora_adaptation_rank,
                lora_adaptation_alpha = self.args.lora_adaptation_alpha,
                lora_adaptation_adapt_attention = self.args.lora_adaptation_adapt_attention,
                lora_adaptation_adapt_mlp = self.args.lora_adaptation_adapt_mlp,
            )

            if "hf:" in self.args.pretrained_checkpoint:
                from huggingface_hub import hf_hub_download

                # url structure: hf:repo:filename
                repo, filename = self.args.pretrained_checkpoint.split(':')[1:]

                local_checkpoint_path = hf_hub_download(repo_id=repo, filename=filename)
                checkpoint = torch.load(local_checkpoint_path, map_location='cpu', weights_only=False)

                checkpoint_model = interpolate_pos_embed(model.backbone, checkpoint)

                print("Load pre-trained checkpoint from HuggingFace: %s" % self.args.pretrained_checkpoint)
                model.backbone.load_state_dict(checkpoint_model)

            elif self.args.pretrained_checkpoint != '':
                checkpoint = torch.load(self.args.pretrained_checkpoint, map_location='cpu', weights_only=False)
                checkpoint_model = checkpoint['model']

                # Check if the model is trained with DINOv2 by checking if the keys contain 'teacher'
                is_dinov2 = any('teacher' in k for k in checkpoint_model.keys())

                if is_dinov2:

                    print("Loading model from DINOv2 training: %s" % self.args.pretrained_checkpoint)

                    # Remove all keys with 'student' in them
                    checkpoint_model = {k: v for k, v in checkpoint_model.items() if 'student' not in k}

                    # Remove 'teacher' from the keys names
                    checkpoint_model = {k.replace('teacher.', ''): v for k, v in checkpoint_model.items()}

                    # Remove all keys with 'dino_loss' in them
                    checkpoint_model = {k: v for k, v in checkpoint_model.items() if 'dino_loss' not in k}

                    # Remove all keys with 'ibot_patch_loss' in them
                    checkpoint_model = {k: v for k, v in checkpoint_model.items() if 'ibot_patch_loss' not in k}

                    # Get all keys with 'backbone' in them and remove 'backbone' from the keys names
                    checkpoint_model = {k.replace('backbone.', ''): v for k, v in checkpoint_model.items() if 'backbone' in k}

                    # Interpolate position embedding
                    checkpoint_model = interpolate_pos_embed(model.backbone, checkpoint_model)

                    model.backbone.load_state_dict(checkpoint_model)

                else:

                    print("Load pre-trained checkpoint from: %s" % self.args.pretrained_checkpoint)

                    # Interpolate position embedding
                    interpolate_pos_embed(model.backbone, checkpoint_model)

                    model.load_state_dict(checkpoint_model)                 

            model.to(self.device)

            model_without_ddp = model
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)

            model_without_ddp = model.module

            return model, model_without_ddp, n_parameters
            
        elif "RETFound" in self.args.model:

            model = models_vit.vit_large_patch16(
                img_size=self.args.input_size,
                num_classes=self.args.nb_classes,
                drop_path_rate=self.args.drop_path,
                global_pool=self.args.global_pool,
            )

            if self.args.RETFound_pretrained and not self.args.eval:
                checkpoint = torch.load(self.args.RETFound_pretrained, map_location='cpu')

                print("Load pre-trained checkpoint from: %s" % self.args.RETFound_pretrained)
                checkpoint_model = checkpoint['model']
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(model, checkpoint_model)

                # load pre-trained model
                msg = model.load_state_dict(checkpoint_model, strict=False)
                print(msg)

                if self.args.global_pool:
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
                else:
                    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

                # manually initialize fc layer
                trunc_normal_(model.head.weight, std=2e-5)

            if self.args.pretrained_checkpoint and self.args.eval:
                checkpoint = torch.load(self.args.pretrained_checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model'])

            model.to(self.device)

            model_without_ddp = model
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            if self.args.fix_backbone == True:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu], find_unused_parameters=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.args.gpu])
            model_without_ddp = model.module

            return model, model_without_ddp, n_parameters
        
    def fix_backbone_(self):

        if self.args.fix_backbone == True:
            for param in self.model.parameters():
                param.requires_grad = False
            if (self.args.block_expansion_positions is not None) and (self.args.pretrained_checkpoint == ''):
                expanded_block_positions = get_expanded_block_positions(self.args.block_expansion_positions)
                print("expanded_block_positions for paramete setting: ", expanded_block_positions)
                print(f"number of block: {len(self.model.module.backbone.blocks)}")
                for position in expanded_block_positions:
                    for param in self.model.module.backbone.blocks[position].parameters():
                        param.requires_grad = True
            for param in self.model.module.head.parameters():
                param.requires_grad = True
                
        return
    
    def get_param_groups_(self):

        if "dinov2" in self.args.model:
            if (self.args.block_expansion_positions is not None) and (self.args.pretrained_checkpoint == ''):
                # Collect parameters from model.head
                head_params = [p for p in self.model.module.head.parameters() if p.requires_grad]
                
                # Collect the IDs of head parameters
                head_param_ids = set(id(p) for p in head_params)

                # Collect the remaining parameters
                other_params = [p for p in self.model.module.parameters() if p.requires_grad and id(p) not in head_param_ids]

                # Create parameter groups
                param_groups = [head_params, other_params]
            else:
                param_groups = filter(lambda p: p.requires_grad, self.model.parameters())

        elif "RETFound" in self.args.model:
            param_groups = lrd.param_groups_lrd(self.model_without_ddp, self.args.weight_decay,
                no_weight_decay_list=self.model_without_ddp.no_weight_decay(),
                layer_decay=self.args.layer_decay
            )
            
        return param_groups
