import imageio
import json
import nltk
import numpy as np
import pandas as pd
import re
import datetime
import requests
import shutil
import sys
import os
import spotipy

from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, redirect, url_for, abort
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
from pathlib import Path
from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from wordcloud import WordCloud
from explicit_words import explicit
from app_credentials import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from afinn import Afinn

# Tensorflow Keras Dependencies
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from tensorflow.keras import backend as K
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# One time run
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC_IMG = os.path.join(APP_ROOT, 'static', 'img')
APP_STATIC_DATA = os.path.join(APP_ROOT, 'static', 'data')


GENIUS_BASE_URL = 'https://api.genius.com'
GENIUS_HEADERS = {'Authorization': 'Bearer ' + GENIUS_API_KEY}

SPOTIFY_CLIENT_CREDENTIALS = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET)
SPOTIFY = spotipy.Spotify(
    client_credentials_manager=SPOTIFY_CLIENT_CREDENTIALS)

STOPS = stopwords.words('english')
FILTER_WORDS = STOPS + explicit  # from explicit_words.py

NRC_LEXICON_SENTIMENT = pd.read_csv(os.path.join(
    APP_STATIC_DATA, 'NRC-Emotion-Lexicon-v0.92.csv'))

AFINNITY_SRC_DF = pd.read_csv(
    os.path.join(
        APP_STATIC_DATA, 'AFINN-111.txt'), names=["word", "score"], sep='\t')

AFINNITY_LIST = AFINNITY_SRC_DF['word'].values.tolist()


global model
global graph
model = Xception(weights='imagenet')
graph = tf.compat.v1.Session()


def get_spotify_track_id(search_string):
    try:
        result = SPOTIFY.search(search_string, limit=1, type='track')
        a = result['tracks']['items'][0]
        return a['id']
    except:
        return "UNKNOWN"

def prepare_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_image_predictions(url_string):
    url = str(url_string)
    data = []
    image_size = (299, 299)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    img = img.resize(image_size, Image.NEAREST)
    image = prepare_image(img)

    # make predictions
    global graph
    with graph.as_default():
        preds = model.predict(image)
        results = decode_predictions(preds)
        for (imagenetID, label, prob) in results[0]:
            row = {"label": label, "probability": float(prob)}
            data.append(row)
        # data['label'] = labels
        # data['probs'] = probs
        return data

# Giving Credits to https://stackoverflow.com/questions/54396405/how-can-i-preprocess-nlp-text-lowercase-remove-special-characters-remove-numb
def preprocess(lyrics_input):
    lyrics = str(lyrics_input)
    lyrics = lyrics.lower()
    lyrics = lyrics.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', lyrics)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in FILTER_WORDS]
    return " ".join(filtered_words)


def build_nrc_sentiment(lyrics_input):
    tokens = RegexpTokenizer(r'\w+').tokenize(lyrics_input)
    lyrics_df = pd.DataFrame(tokens, columns=['Word'])
    lyrics_df = lyrics_df.merge(NRC_LEXICON_SENTIMENT, how="inner", on="Word")
    lyrics_df = lyrics_df.drop(columns=["Word"])
    results_df = pd.DataFrame(lyrics_df.sum(), columns=['Weight'])
    results_df.sort_values(by=['Weight'], ascending=True, inplace=True)
    final = {}
    final['Sentiment'] = list(results_df.to_dict()['Weight'].keys())
    final['Weight'] = list(results_df.to_dict()['Weight'].values())
    return final


def build_cloud(lyrics_input):
    mask_image = imageio.imread(os.path.join(
        APP_STATIC_IMG, 'mask_circle.png'))
    wordcloud = WordCloud(width=500, height=500, colormap='tab20',
                          mask=mask_image, background_color=None, mode="RGBA")
    wordcloud = wordcloud.generate(lyrics_input)
    shutil.rmtree(os.path.join(APP_STATIC_IMG, 'output'), ignore_errors=True)
    time_now = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y%m%d%H%M%S")
    filepath = os.path.join(APP_STATIC_IMG, 'output', f'cloud_{time_now}.png')
    os.mkdir(os.path.join(APP_STATIC_IMG, 'output'))
    wordcloud = wordcloud.to_file(filepath)
    return f'cloud_{time_now}.png'


def build_wordcount(lyrics_input):
    wordcloud_dict = {}
    song_lyrics = TextBlob(lyrics_input)
    lyrics = song_lyrics.word_counts.items()
    lyrics = [lyric for lyric in lyrics]
    sorted_lyrics = sorted(lyrics, key=itemgetter(1), reverse=True)
    top20_lyrics = sorted_lyrics[1:21]
    wordcloud_dict['Word'] = [a[0] for a in top20_lyrics]
    wordcloud_dict['Count'] = [a[1] for a in top20_lyrics]
    # df_lyrics = pd.DataFrame(top20_lyrics, columns=['Words', 'Count'])
    # print(wordcloud_dict)
    return wordcloud_dict

def get_tags_for_song(song_name, artist_name):
    API_KEY = "a6627484c312d94e547aeac4f28b739d"
    artist = artist_name
    artist = artist.replace(' ',"+")
    artist = artist.replace("''","")
    song = song_name
    song = song.replace(' ',"+")
    song = song.replace(",","")
    try:
        query_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={API_KEY}&artist={artist}&track={song}&format=json"
        response = requests.get(query_url)
        tags = response.json()['track']['toptags']['tag']
        tag = []
        for i in range(len(tags)):
            if tags[i]['name'] == artist.lower():
                continue
            else:
                tag.append(tags[i]['name'])
        if len(tag) == 0:
            return "Not Enough Data On This Songs"
        else:
            return tag
    except:
        return "Something Went Wrong"

def build_vader(lyrics_input):
    analyzer = SentimentIntensityAnalyzer()
    text_sentiment = analyzer.polarity_scores(lyrics_input)
    #Split the dictionary for graphing
    compound_score = text_sentiment.pop('compound')
    lemmatized_tokens = nltk.word_tokenize(lyrics_input)
    common = list(set(AFINNITY_LIST) & set(lemmatized_tokens))
    common_string = ', '.join([str(x) for x in common])
    text_sentiment1 = {}
    text_sentiment1['Sentiment'] = ["Negative", "Neutral", "Positive"]
    text_sentiment1['Weight'] = list(text_sentiment.values())
    text_sentiment1['affinity_words'] = common_string
    text_sentiment1['compound_score'] = compound_score

    return text_sentiment1



def get_song_details(song_id, song_search):
    song_dict = {}
    # Get song metadata
    search_url = GENIUS_BASE_URL + '/songs/' + str(song_id)
    response = requests.get(search_url, headers=GENIUS_HEADERS)
    song = response.json()
    song_dict['id'] = song['response']['song']['id']
    song_dict['title'] = song['response']['song']['title']
    song_dict['artist'] = song['response']['song']['primary_artist']['name']
    song_dict['artist_image'] = song['response']['song']['primary_artist']['image_url']
    song_dict['url'] = song['response']['song']['url']
    song_dict['album'] = song['response']['song']['album']['name']
    song_dict['year'] = song['response']['song']['release_date_for_display']
    song_dict['song_image'] = song['response']['song']['song_art_image_thumbnail_url']

    spotify_id = [a['native_uri'] for a in song['response']['song']['media'] if a['provider'] == 'spotify']
    spotify_id = spotify_id[0].replace('spotify:track:', '') if len(spotify_id) >= 1 else 'UNKNOWN'
    spotify_id = get_spotify_track_id(song_search) if (spotify_id == 'UNKNOWN') else spotify_id

    song_dict['spotify_id'] = spotify_id

    # Get song lyrics
    response = requests.get(
        song['response']['song']['url'], headers=GENIUS_HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    lyrics = soup.find('div', class_='lyrics').get_text()
    lyrics = lyrics.replace('\n', ' ').replace('\r', ' ')
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    song_dict['lyrics'] = lyrics
    # print(song_dict)
    return song_dict


@app.route("/results", methods=["GET", "POST"])
def results():
    try:
        print(request.values)
        if request.method == "POST":
            search_string = request.form["song_search"]
            print("Getting song details...")
            song_data_dict = get_song_details(request.form["song_id"], search_string)
            print(search_string)
            cleaned_lyrics = preprocess(song_data_dict["lyrics"])
            print("Processing wordcount...")
            bar_data = build_wordcount(cleaned_lyrics)
            print("Processing wordcloud...")
            wordcloud_path = build_cloud(cleaned_lyrics)
            print("Processing NRC Sentiment...")
            nrc_sentiment = build_nrc_sentiment(cleaned_lyrics)
            print("Processing Vader Sentiment...")
            vader_sentiment = build_vader(cleaned_lyrics)
            print("Getting Song Tags...")
            song_tags = get_tags_for_song(song_data_dict["title"], song_data_dict["artist"])
            print("Getting Image Predictions...")
            print(song_data_dict["song_image"])
            image_predictions = get_image_predictions(song_data_dict["song_image"])
            print(image_predictions)
            print("Returning results")
            return render_template("base.html", song_data=song_data_dict, bar_data=bar_data, wordcloud_path=wordcloud_path, nrc_sentiment=nrc_sentiment, vader_sentiment=vader_sentiment, song_tags=song_tags, image_predictions=image_predictions)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return redirect(url_for("index", json_content={'Lyrics Not Found'}, message_type="Error", message_content="Lyrics Not Found"))


@app.route("/_autocomp/<search>")
def autocomplete(search):
    query_string = search

    search_url = GENIUS_BASE_URL + '/search'
    data = {'q': query_string}
    response = requests.get(search_url, data=data, headers=GENIUS_HEADERS)
    json = response.json()
    song_results = []

    for hit in json['response']['hits']:
        song_dict = {}
        song_dict['TitleArtist'] = hit['result']['full_title']
        song_dict['search'] = hit['result']['title'] + \
            ' ' + hit['result']['primary_artist']['name']
        song_dict['songID'] = hit['result']['id']
        song_results.append(song_dict)

    return jsonify(song_results)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html', json_content={'Error 404!': 'Page not found!'}, message_type="Error", message_content="WHERE WERE YOU GOING? :/"), 404


@app.errorhandler(400)
def invalid_parameter(e):
    return render_template('index.html', json_content={'Error!': 'Invalid parameter!'}, message_type="Error", message_content="Invalid parameter passed!"), 400


@app.route("/", methods=['GET', 'POST'])
def index():
    json_content = request.args.get('json_content') or ""
    message_type = request.args.get('message_type') or ""
    message_content = request.args.get('message_content') or ""
    return render_template('index.html', json_content=json_content, message_type=message_type, message_content=message_content)


if __name__ == "__main__":
    app.run(debug=True)
