import streamlit as st
import cv2
import time
from ultralytics import YOLO
from PIL import Image

confidence = .25

model = YOLO("runs/best.pt")


def result_from_model(model, img):
    results = model.predict(img)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                          (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img


def image_input():
    img_file = None

    img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "uploaded_data/image/image." + img_bytes.name.split('.')[-1]
        Image.open(img_bytes).save(img_file)

    if img_file:
        img_raw = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = img_raw.copy()
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Selected Image")
        with col2:
            img = result_from_model(model, img)
            st.image(img, caption="Model prediction")


def video_input():
    vid_file = None

    vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        vid_file = "uploaded_data/video/video." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = result_from_model(model, frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()


# Set Streamlit app title and description
st.title("Animal Detection with YOLOv8")
st.markdown("Detect wolves, jaguars, humpback whales, rhinos, and crocodiles.")

# Upload an image for detection

st.sidebar.header("Upload Image or Video")
options = st.sidebar.radio(
    "Set selectbox label visibility 👉",
    key="visibility",
    options=["Image", "Video"],
)

if options == "Image":
    image_input()
else:
    video_input()

# Display some information about the project
st.sidebar.header("About")
st.sidebar.info("This project uses YOLOv8 for animal detection.")

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created by Mert Akgül")

# You can add more features, explanations, and styling as needed.
