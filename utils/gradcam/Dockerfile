FROM tensorflow/tensorflow:2.4.1-gpu

RUN pip3 install pandas numpy pillow sklearn keras tensorflow-gpu tqdm
RUN pip3 install --upgrade protobuf
COPY extract_max_confidence.py /extract_max_confidence.py
COPY backbone.py /backbone.py

CMD ["python3", "./extract_max_confidence.py"]