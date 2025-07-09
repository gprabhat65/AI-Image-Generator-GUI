import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch 
from torch import autocast
from diffusers import StableDiffusionPipeline

#Create the app 

app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master= app, height=40, width= 512, font=("Arial", 20), text_color = "black", fg_color = "white" )
prompt.place(x = 10, y = 10)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant= "fp16", torch_dtype=torch.float16, use_auth_token = auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5).image[0]

    img = ImageTk.PhotoImage(image)
    img.save('GeneratedImage.png')
    lmain.configure(image = img)



trigger = ctk.CTkButton(master=app, height = 40, width = 120, font=("Arial", 20), text_color = "white", fg_color = "blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

lmain = ctk.CTkLabel(master=app, height = 512, width=512)
lmain.place(x=10, y=120)

app.mainloop()

































# import tkinter as tk
# import customtkinter as ctk

# from PIL import ImageTk
# from authtoken import auth_token

# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline

# # Create the main app window
# app = ctk.CTk()
# app.geometry("532x622")
# app.title("Stable Bud")
# ctk.set_appearance_mode("dark")

# # Prompt Entry
# prompt = ctk.CTkEntry(
#     master=app, height=40, width=512, font=("Arial", 20),
#     text_color="black", fg_color="white"
# )
# prompt.place(x=10, y=10)

# # Image Display Label
# lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
# lmain.place(x=10, y=60)

# # Generate Button
# trigger = ctk.CTkButton(
#     master=app, height=40, width=120, font=("Arial", 20),
#     text="Generate", text_color="white", fg_color="blue"
# )
# trigger.place(x=10, y=580)

# # Run the app
# app.mainloop()
