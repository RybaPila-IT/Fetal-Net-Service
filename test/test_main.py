import json
import os

from fastapi.testclient import TestClient
from fastapi import status
from main import app, ACCESS_TOKEN_ENV_KEY

main_url = '/'
predict_url = '/predict'
access_token = 'access_token'
client = TestClient(app)

os.environ[ACCESS_TOKEN_ENV_KEY] = access_token


def test_main_request():
    resp = client.get(url=main_url)
    resp_body = json.loads(resp.content.decode())

    assert resp.status_code == status.HTTP_200_OK
    assert resp_body['message'] is not None


def test_unauthorized_request():
    resp = client.post(
        url=predict_url,
        json={
            'photo': 'Random pixels',
            'attributes': {
                'attribute_1': 'value'
            }
        }
    )

    assert resp.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_access_token_request():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': 'Bearer invalid_token'},
        json={
            'photo': 'Random pixels',
            'attributes': {
                'attribute_1': 'value'
            }
        }
    )

    assert resp.status_code == status.HTTP_403_FORBIDDEN


def test_invalid_encoded_pixels_request():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={
            'photo': 'Random pixels',
            'attributes': {
                'attribute_1': 'value'
            }
        }
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_invalid_image_request_missing_image_size():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={
            'photo': 'aGVsbG8gd29ybGQh',
            'attributes': {
                'pixel_spacing': 0.254439
            }
        }
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_invalid_image_request_missing_pixel_spacing():
    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={
            'photo': 'aGVsbG8gd29ybGQh',
            'attributes': {
                'image_size': [975, 743]
            }
        }
    )

    assert resp.status_code == status.HTTP_400_BAD_REQUEST


def test_valid_request():
    with open('test/data/samples/encoded/0427_head_203.txt') as f:
        encoded_pixels = f.read()

    resp = client.post(
        url=predict_url,
        headers={'Authorization': f'Bearer {access_token}'},
        json={
            'photo': encoded_pixels,
            'attributes': {
                'image_size': [975, 743],
                'pixel_spacing': 0.254439
            }
        }
    )
    resp_body = json.loads(resp.content.decode())

    assert resp.status_code == status.HTTP_200_OK
    assert resp_body.get('prediction') is not None
    assert resp_body.get('prediction').get('body_part') == 'head'
    assert resp_body.get('photo') is not None
