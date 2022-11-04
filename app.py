import gradio as gr
import requests

# GPT-J-6B API
API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
headers = {"Authorization": "Bearer ******"}
prompt = """
me: I'm Twimbly Twombly. I live in California. I'm a robot."""

prev_chat = """
you: Hello. How are you?
me: I'm fine"""

examples = [["how are you?"], ["hello"]]


def chat_generate(word):
#   print(f"**current reply")
  global prev_chat
  p = prompt + prev_chat + "\n" + "you: " + word.lower() + "\n" + "me: "
  print(f"*****Inside chat_generate - Prompt is :{p}")
  json_ = {"inputs": p,
            "parameters":
            {
            "top_p": 0.9,
          "temperature": 1.1,
          "max_new_tokens": 50,
          "return_full_text": False
          }}
  response = requests.post(API_URL, headers=headers, json=json_)
  output = response.json()
  output_tmp = output[0]['generated_text']
  reply = output_tmp.split("you:")[0] # +"."
  print(f"Chat Response being returned is: {reply}")
  prev_chat = "you: " + word.lower() + "\n" + "me: " + reply
  return reply

def text_to_image(reply):
  print("*****Inside Text_to_image")
  reply = " ".join(reply.split('\n'))
  reply = reply + " oil on canvas."
  steps, width, height, images, diversity = '50','256','256','1',15
  img = gr.Interface.load("spaces/multimodalart/latentdiffusion")(reply, steps, width, height, images, diversity)[0]
  return img

demo = gr.Blocks()

with demo:
  gr.Markdown("<h1><center>Twimbly Twombly</center></h1>")
  gr.Markdown(
        "<div>Hi I'm Twimbly Twombly ready to talk to you.</div>"
    )
  input_word = gr.Textbox(placeholder="Enter a word here to chat..")
  chat_txt = gr.Textbox(lines=1)
  # output_image = gr.Image(type="filepath", shape=(256,256))
  
  b1 = gr.Button("Send")
  #b2 = gr.Button("Imagine")

  b1.click(chat_generate, input_word, chat_txt)
  #b2.click(text_to_image, chat_txt, output_image)
  #examples=examples

demo.launch(enable_queue=True, debug=True)