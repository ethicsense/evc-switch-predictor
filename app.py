import gradio as gr
import os
import cv2
import model_selector
import argparse


def get_model(model_type, model_name):
    mg =  model_selector.model_generator(model_type, model_name)
    model = mg.get_model()
    if model:
        out = "Success"
    else:
        out = "Fail"

    return out, model



def get_weights(f):

    return f


def run_model():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_name',
        type=str,
        default='0.0.0.0'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860
    )
    args=parser.parse_args()


    with gr.Blocks() as demo:
        radio = gr.Radio(label="Model Class", choices=["ResNet", "EfficientNet", "MobileNet", "YOLOv8"])
        model_list = gr.Dropdown(label="Models Available", choices=[], interactive=True)
        w_list = os.listdir("weights/")
        model_map = {
            "YOLOv8" : w_list
        }

        def filter_models(choice):
            if choice in model_map.keys():
                return gr.Dropdown(
                    choices=model_map[choice], value=model_map[choice][1]
                )
            else:
                return gr.Dropdown(visible=False)
        
        radio.change(filter_models, inputs=radio, outputs=model_list)
        b1 = gr.Button(value="Load Model")

        with gr.Row() as output_row:
            txt1 = gr.Textbox(label="Done or Not")
            txt2 = gr.TextArea(label="Model Information")

        b1.click(get_model, inputs=[radio, model_list], outputs=[txt1, txt2])

        b2 = gr.Button(value="Start Prediction")



    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        debug=True
    )