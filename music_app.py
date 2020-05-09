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




app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC_IMG = os.path.join(APP_ROOT, 'static','img')
APP_STATIC_DATA = os.path.join(APP_ROOT, 'static','data')

GENIUS_API_KEY = 'iPRpGvyPEPwHqexfQo75LsL0i2pPQxhkw-P5WStYbdvmUq-PQyf7ppCnT92Z-ZQc'
genius_base_url = 'https://api.genius.com'
headers = {'Authorization': 'Bearer ' + GENIUS_API_KEY}

nrc_df = pd.read_csv(os.path.join(APP_STATIC_DATA, 'NRC-Emotion-Lexicon-v0.92.csv'))

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
    filtered_words = [w for w in tokens if len( w) > 2 if not w in stopwords.words('english')]
    return " ".join(filtered_words)

def build_nrc_sentiment(lyrics_input):
    tokens=RegexpTokenizer(r'\w+').tokenize(lyrics_input)
    lyrics_df=pd.DataFrame(tokens, columns=['Word'])
    lyrics_df=lyrics_df.merge(nrc_df, how="inner", on="Word")
    lyrics_df=lyrics_df.drop(columns=["Word"])
    results_df = pd.DataFrame(lyrics_df.sum(), columns=['Weight'])
    results_df.sort_values(by=['Weight'], ascending=True, inplace=True)
    final = {}
    final['Sentiment'] = list(results_df.to_dict()['Weight'].keys())
    final['Weight'] = list(results_df.to_dict()['Weight'].values())
    return final

def build_cloud(lyrics_input):
    mask_image = imageio.imread(os.path.join(APP_STATIC_IMG, 'mask_circle.png'))
    wordcloud = WordCloud(width=500, height=500, colormap='tab20', mask=mask_image, background_color=None, mode="RGBA")
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
    print(wordcloud_dict)
    return wordcloud_dict

def get_song_details(song_id):
    song_dict = {}

    # Get song metadata
    search_url = genius_base_url + '/songs/' + str(song_id)
    response = requests.get(search_url, headers=headers)
    song = response.json()
    song_dict['id'] = song['response']['song']['id']
    song_dict['title'] = song['response']['song']['title']
    song_dict['artist'] = song['response']['song']['primary_artist']['name']
    song_dict['artist_image'] = song['response']['song']['primary_artist']['image_url']
    song_dict['url'] = song['response']['song']['url']
    song_dict['album'] = song['response']['song']['album']['name']
    song_dict['year'] = song['response']['song']['release_date_for_display']
    song_dict['song_image'] = song['response']['song']['song_art_image_thumbnail_url']
    spotify_uri = [a['native_uri'] for a in song['response']['song']['media'] if a['provider'] == 'spotify']
    song_dict['spotify_uri'] = spotify_uri[0] if len(spotify_uri) >= 1 else 'UNKNOWN'
    song_dict['spotify_id'] = spotify_uri[0].replace('spotify:track:', '') if len(spotify_uri) >= 1 else 'UNKNOWN'

    # Get song lyrics
    response = requests.get(song['response']['song']['url'], headers=headers)
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
            song_data_dict = get_song_details(request.form["song_id"])
            cleaned_lyrics=preprocess(song_data_dict["lyrics"])
            bar_data = build_wordcount(cleaned_lyrics)
            wordcloud_path=build_cloud(cleaned_lyrics)
            nrc_sentiment = build_nrc_sentiment(cleaned_lyrics)
            print(bar_data)
            print(nrc_sentiment)
            return render_template("base.html", song_data=song_data_dict, bar_data=bar_data, wordcloud_path=wordcloud_path, nrc_sentiment=nrc_sentiment)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return redirect(url_for("index", json_content={'Lyrics Not Found'}, message_type="Error", message_content="Lyrics Not Found"))



@app.route("/_autocomp/<search>")
def autocomplete(search):
    query_string = search
    
    search_url = genius_base_url + '/search'
    data = {'q': query_string}
    response = requests.get(search_url, data=data, headers=headers)
    json = response.json()
    song_results = []

    for hit in json['response']['hits']:
        song_dict = {}
        song_dict['TitleArtist'] = hit['result']['full_title']
        song_dict['URL'] = hit['result']['url']
        song_dict['songID'] = hit['result']['id']
        song_results.append(song_dict)

    return jsonify(song_results)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html',json_content={ 'Error 404!' : 'Page not found!'}, message_type="Error", message_content="WHERE WERE YOU GOING? :/"), 404

@app.errorhandler(400)
def invalid_parameter(e):
    return render_template('index.html',json_content={ 'Error!' : 'Invalid parameter!'}, message_type="Error", message_content="Invalid parameter passed!"), 400


@app.route("/", methods=['GET', 'POST'])
def index():
    json_content = request.args.get('json_content') or ""
    message_type = request.args.get('message_type') or ""
    message_content = request.args.get('message_content') or ""
    return render_template('index.html', json_content=json_content, message_type=message_type, message_content=message_content)

if __name__ == "__main__":
    app.run(debug=True)
