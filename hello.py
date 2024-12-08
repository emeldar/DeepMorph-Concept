from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import torchvision.transforms as transforms
from transformers.utils import logging
from torch.utils.data import TensorDataset, DataLoader

def slerp(low, high, val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def preprocess_image(image_path, target_size=(512, 512)):
    # Load the image using Pillow
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),  # Resize to 512x512
        transforms.ToTensor(),  # Convert image to tensor (C, H, W) in range [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    image_tensor = transform(image)

    return image_tensor

def batch_encode_images(image_tensors, encoder):
    # Stack tensors along a new dimension to create a batch
    batched_images = torch.stack(image_tensors)

    # Encode the batch of images into latent space
    with torch.no_grad():
        latent_space = encoder(batched_images).latent_dist.sample()

    return latent_space

def interpolate_latent_gif(latent_start, latent_end, frame_count, interpolation_method=torch.lerp):
    image_list = [latent_start]

    for i in range(frame_count):
        if i == 0:
            continue
        if i == frame_count-1:
            image_list.append(latent_end)
        interpolated_image = interpolation_method(latent_start, latent_end, i/(frame_count-1))
        image_list.append(interpolated_image)

    return torch.stack(image_list)
    
def interpolate_latent_waypoints(waypoints, frame_counts, device, interpolation_method=torch.lerp):
    image_list = []

    for frame_idx in range(len(frame_counts)):
        waypoint_frames = frame_counts[frame_idx]
        for i in range(waypoint_frames):
            if i == waypoint_frames-1:
                continue
            interpolated_image = interpolation_method(waypoints[frame_idx], waypoints[frame_idx+1], i/(waypoint_frames-1))
            interpolated_image = torch.add(interpolated_image, torch.div(generate_random_latent().to(device), 5))
            image_list.append(interpolated_image)

    image_list.append(waypoints[-1])

    return torch.stack(image_list)

    # for i in range(frame_count):
    #     if i == 0:
    #         continue
    #     if i == frame_count-1:
    #         image_list.append(latent_end)
    #     interpolated_image = interpolation_method(latent_start, latent_end, i/(frame_count-1))
    #    image_list.append(interpolated_image)

    #return torch.stack(image_list)

def convert_output_to_pillow(image_tensor):

    # Convert the image tensor (in [-1, 1]) to a NumPy array (in [0, 255])
    #image_array = image_tensor.squeeze().cpu().numpy()
    postprocess_transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Scale from [-1, 1] to [0, 1]
        transforms.Lambda(lambda x: x.clamp(0, 1)),  # Clip values to [0, 1]
        transforms.ToPILImage()  # Convert tensor to PIL image
    ])

    return postprocess_transform(image_tensor.cpu())

def generate_random_latent():
    return torch.sub(torch.mul(torch.rand(4, 64, 64), 20), 10)

def main():
    model_id = "CompVis/stable-diffusion-v1-4"
    #model_id = "nitrosocke/spider-verse-diffusion"

    logging.set_verbosity_warning()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    print("Starting Morph!")

    image1 = preprocess_image('./images/image.png').to(device)
    image2 = preprocess_image('./images/bucket.png').to(device)
    #image3 = preprocess_image('./images/dr_shrimp.png').to(device)

    #print("Loaded Images!")

    #batched_images = torch.stack([image1, image2, image3])
    batched_images = torch.stack([image1, image2])

    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    vae = pipeline.vae

    print("Encoding Seed Images!")

    with torch.no_grad():
        latent_space_images = vae.encode(batched_images).latent_dist.sample()

    #latent1 = generate_random_latent().to(device)
    #latent_bucket = latent_space_images[0]
    
    print("Interpolating Latent Space!")
    #latent_gif = interpolate_latent_gif(latent_space_images[0], latent_space_images[1], 20, slerp)

    #latent_space_images = torch.stack([latent1, latent_bucket])

    latent_gif = interpolate_latent_waypoints(latent_space_images, [120], device)

    gif_dataset = TensorDataset(latent_gif)
    gif_dataloader = DataLoader(gif_dataset, batch_size=10)

    print("Decoding GIF Images!")

    decoded_images = []

    for batch_idx, (batch,) in enumerate(gif_dataloader):
        torch.cuda.empty_cache()
        print(f"Decoding batch {batch_idx+1} of {len(gif_dataloader)}!")
        with torch.no_grad():
            decoded_image_tensors = vae.decode(batch).sample.cpu()
            decoded_images.append(decoded_image_tensors)

    decoded_images = torch.cat(decoded_images, 0)

    print("Saving GIF!")

    gif_images = [convert_output_to_pillow(image_tensor) for image_tensor in decoded_images]

    gif_images[0].save(
        './outputs/morphed.gif',
        save_all=True,
        append_images=gif_images[1:],
        format='GIF',
        duration=4000/120,
        loop=0
    )

    print("Done!")

    #image1_reconstructed = convert_output_to_pillow(decoded_image_tensors[0])
    #image1_reconstructed.save('./outputs/image_reconstructed.png')

    #image2_reconstructed = convert_output_to_pillow(decoded_image_tensors[1])
    #image2_reconstructed.save('./outputs/bucket_reconstructed.png')

    #image_morph_output = convert_output_to_pillow(decoded_image_tensors[2])
    #image_morph_output.save('./outputs/morph_output.png')

# def main_old():
#     model_id = "CompVis/stable-diffusion-v1-4"

#     logging.set_verbosity_warning()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using device:', device)
#     print()

#     #Additional Info when using cuda
#     if device.type == 'cuda':
#         print(torch.cuda.get_device_name(0))
#         print('Memory Usage:')
#         print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#         print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

#     print("Starting Morph!")

#     pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
#     vae = pipeline.vae
#     text_encoder = pipeline.text_encoder
#     tokenizer = pipeline.tokenizer

#     prompt1 = "dr phil eating a sandwich"
#     text_input1 = tokenizer(prompt1, return_tensors="pt").input_ids.to("cuda")
#     prompt2 = "steve harvey playing the recorder"
#     text_input2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")

#     print(text_input1.shape, text_input2.shape)

#     print("Encoding Seed Images!")

#     with torch.no_grad():
#         text_embedding1 = text_encoder(text_input1)[0]
#         text_embedding2 = text_encoder(text_input2)[0]

#     print(text_embedding1.shape, text_embedding2.shape)
    
#     print("Interpolating Latent Space!")
#     #latent_gif = interpolate_latent_gif(text_embedding1, text_embedding2, [], slerp)

#     latent_space_images = torch.stack([text_embedding1, text_embedding2])

#     latent_gif = interpolate_latent_waypoints(latent_space_images, [1000], device)

#     gif_dataset = TensorDataset(latent_gif)
#     gif_dataloader = DataLoader(gif_dataset, batch_size=100)

#     print("Decoding GIF Images!")

#     decoded_images = []

#     scheduler = pipeline.scheduler
#     timestep = torch.tensor([scheduler.config.num_train_timesteps - 1], device="cuda")

#     for batch_idx, (batch,) in enumerate(gif_dataloader):
#         torch.cuda.empty_cache()
#         print(f"Decoding batch {batch_idx+1} of {len(gif_dataloader)}!")
#         with torch.no_grad():
#             image = pipeline(prompt=None, num_inference_steps=1, guidance_scale=7.5, prompt_embeds=batch[0])
#             decoded_images.append(image.images[0])

#     #decoded_images = torch.cat(decoded_images, 0)

#     print("Saving GIF!")

#     #gif_images = [convert_output_to_pillow(image_tensor) for image_tensor in decoded_images]

#     decoded_images[0].save(
#         './outputs/morphed.gif',
#         save_all=True,
#         append_images=decoded_images[1:],
#         format='GIF',
#         duration=10000/1000,
#         loop=0
#     )

#     print("Done!")

#     #image1_reconstructed = convert_output_to_pillow(decoded_image_tensors[0])
#     #image1_reconstructed.save('./outputs/image_reconstructed.png')

#     #image2_reconstructed = convert_output_to_pillow(decoded_image_tensors[1])
#     #image2_reconstructed.save('./outputs/bucket_reconstructed.png')

#     #image_morph_output = convert_output_to_pillow(decoded_image_tensors[2])
#     #image_morph_output.save('./outputs/morph_output.png')

if __name__ == "__main__":
    main()
