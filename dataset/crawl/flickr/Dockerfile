FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install PyDictionary flickrapi pandas

COPY ./flickr_crawler.py ./flickr_crawler.py

CMD [ "python", "./flickr_crawler.py" ]