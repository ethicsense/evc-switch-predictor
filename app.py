import gradio as gr
import os
import cv2
import model_manager as mm
import argparse
import datetime
import numpy as np
from PIL import Image


def get_model(model_type, model_name):
    mg =  mm.model_generator(model_type, model_name)
    model = mg.get_model()
    if model:
        out = "Success"
    else:
        out = "Fail"

    return out, model


def new_weights(f):

    return f


def run_model(video, model):
    output_video = model.predict(video, model)
    
    return output_video



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
        radio = gr.Radio(label="Model Class", choices=["YOLOv8"])
        model_list = gr.Dropdown(label="Models Available", choices=[], interactive=True)
        w_list = os.listdir("weights")
        model_map = {
            "YOLOv8" : w_list
        }

        def filter_models(choice):
            if choice in model_map.keys():
                return gr.Dropdown(
                    choices=model_map[choice], value=model_map[choice][2]
                )
            else:
                return gr.Dropdown(visible=False)
        
        radio.change(filter_models, inputs=radio, outputs=model_list)
        b1 = gr.Button(value="Load Model")

        with gr.Row() as output_row:
            txt1 = gr.Textbox(label="Done or Not")
            txt2 = gr.TextArea(label="Model Information")

        mode = gr.Radio(label="Select Mode", choices=["video", "stream"])
        url = gr.Textbox(
            label="enter RTSP url",
            value="rtsp://user1:ketiabcs@evc.re.kr:39091/h264Preview_01_main"
        )
        b2 = gr.Button(value="Start Prediction")

        out_frames = gr.Image(
            label="last frame",
            source="webcam",
            streaming=True
        )
        with gr.Row() as output_row:
            out_video = gr.Video()

        def submit_pred(radio, model_list, inp_video):
            tmp, model = get_model(radio, model_list)
            del tmp
            output = mm.predict(inp_video, model, mode)

            return output

        def stream_infer(radio, model_list, url):
            tmp, model = get_model(radio, model_list)
            del tmp

            currenttime = datetime.datetime.now()
            video_capture = cv2.VideoCapture(url)
            
            video_capture.set(3, 100)
            video_capture.set(4, 100)
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))

            streaming_window_width = int(video_capture.get(3))
            streaming_window_height = int(video_capture.get(4))

            filename = str(currenttime.strftime('%Y %m %d %H %M %S'))
            path = os.getcwd() + "/stream/" + f"{filename}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (streaming_window_width, streaming_window_height))

            iterating, frame = video_capture.read()
            frame_cnt = 0
            imgs_np_list = []

            while iterating:
                results = model(frame)
                frame_cnt += 1
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                out.write(annotated_frame)
                imgs_np_list.append(annotated_frame)

                if frame_cnt >= 30:
                    for i in imgs_np_list[-30:] :
                        yield i, None
                        frame_cnt = 0

                iterating, frame = video_capture.read()

            video_capture.release()
            yield annotated_frame, path


        b1.click(get_model, inputs=[radio, model_list], outputs=[txt1, txt2])
        b2.click(
            stream_infer,
            [radio, model_list, url],
            [out_frames, out_video]
        )



    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        debug=True,
    )