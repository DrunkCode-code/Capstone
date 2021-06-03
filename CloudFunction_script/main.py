import logging
from operator import itemgetter
import os

from flask import jsonify
from flask import Flask, request
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import requests
import json
import tensorflow as tf
# lebar gambar
lebar = 150
# label prediksi
label = ['Anggur','Brokoli','Gandum','Jagung','Kapas','Kubis','Padi','Rami','Tebu','Zaitun']

# simpan endpoint
ai_platform_url = aiplatform.gapic.PredictionServiceClient(client_options={
    'api_endpoint': 'us-central1-aiplatform.googleapis.com'
})
aip_endpoint = f'projects/{os.environ["GCP_PROJECT"]}/locations/us-central1/endpoints/{os.environ["ENDPOINT_ID"]}'


def get_prediction(instance):
    logging.info('Sedang mengirim request prediksi ke AI Platform...')
    # coba request ke model Vertex AI
    try:
        json_instance = json_format.ParseDict(instance, Value())
        response = ai_platform_url.predict(endpoint=aip_endpoint,
                                      instances=[json_instance])
        return list(response.predictions[0])
    # gagal request
    except Exception as err:
        logging.error(f'Prediksi gagal dengan keterangan: {type(err)}: {err}')
        return None


def preprocess_image(image_url):
    logging.info(f'Fetching request gambar: {image_url}')
    # coba request gambar
    try:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        assert image_response.headers.get('Content-Type') == 'image/jpeg'
    # gagal fetching
    except (ConnectionError, requests.exceptions.RequestException,
            AssertionError):
        logging.error(f'Error fetching gambar: {image_url}')
        return None

    logging.info('Sedang decoding dan preprocessing gambar ...')
    # resize gambar
    image = tf.io.decode_jpeg(image_response.content, channels=3)
    image = tf.image.resize_with_pad(image, lebar, lebar)
    image = image / 255.
    # membuat gambar menjadi json
    return image.numpy().tolist()


def predict(request):
    # HTTP Request method harus POST
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Jika bukan post
    if request.method != 'POST':
        return ('Tidak ditemukan!', 404)

    # ubah nilai header ke main request
    headers = {'Access-Control-Allow-Origin': '*'}

    # jika bukan json atau 'image_url' tidak ditemukan di json
    request_json = request.get_json(silent=True)
    if not request_json or not 'image_url' in request_json:
        return ('Kesalahan Request', 400, headers)

    # jika instance gagal melakukan preprocess
    instance = preprocess_image(request_json['image_url'])
    if not instance:
        return ('Kesalahan Request', 400, headers)

    # lakukan prediksi, jika gagal maka error 500
    coba_prediksi = get_prediction(instance)
    if not coba_prediksi:
        return ('Gagal memprediksi!', 500, headers)

    probabilities = zip(label, coba_prediksi)
    # max_probabilities = max(probabilities)
    sorted_probabilities = sorted(probabilities, key=itemgetter(1), reverse=True)
    # ambil hanya key tertinggi
    max_probabilities = sorted_probabilities[0][0]
    respon_json = {
        "prediction": str(max_probabilities)
    }
    
    logging.info('Prediksinya adalah %s', respon_json)
    # return (jsonify(max_probabilities), 200, headers)
    # kirim response dalam bentuk json
    return json.dumps(respon_json)

