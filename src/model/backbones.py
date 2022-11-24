import timm
import torchvision
import torch


def build_timm_model(model_name, num_classes=1000, pretrained=False,
                     checkpoint_path=None):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def build_convnext_large(num_classes=1000, pretrained=False, checkpoint_path=None,
                   create_prototypes=False):
    model =  build_timm_model('convnext_large', num_classes=num_classes,
                              pretrained=pretrained, checkpoint_path=checkpoint_path)

    if create_prototypes:
        del model.head.fc

    return model


if __name__ == "__main__":
    model = build_convnext_large(num_classes=64, pretrained=False,
                           checkpoint_path=None, create_prototypes=False)
    print(model)
    print('Output shape', model(torch.randn(1, 3, 224, 224)).shape)
