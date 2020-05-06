from wordcloud import WordCloud
import imageio
import sys
import numpy as np
import requests
import json
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask import abort
from bs4 import BeautifulSoup


from textblob import TextBlob
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from wordcloud import WordCloud
from PIL import Image

app = Flask(__name__)

GENIUS_API_KEY = 'iPRpGvyPEPwHqexfQo75LsL0i2pPQxhkw-P5WStYbdvmUq-PQyf7ppCnT92Z-ZQc'
genius_base_url = 'https://api.genius.com'
headers = {'Authorization': 'Bearer ' + GENIUS_API_KEY}

# nltk.download('stopwords')
stops = stopwords.words('english')



def build_cloud(lyrics_input):
    mask_image = imageio.imread('static\img\mask_circle.png')
    wordcloud = WordCloud(width=500, height=500, colormap='prism', mask=mask_image, background_color=None, mode="RGBA")
    wordcloud = wordcloud.generate(lyrics_input)
    wordcloud = wordcloud.to_file('static\img\cloud.png')

def build_wordcloud(lyrics_input):
    wordcloud_dict = {}
    song_lyrics = TextBlob(lyrics_input)
    lyrics = song_lyrics.word_counts.items()
    lyrics = [lyric for lyric in lyrics if lyric[0] not in stops]
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
    song_dict['year'] = song['response']['song']['release_date']
    song_dict['song_image'] = song['response']['song']['song_art_image_thumbnail_url']

    # Get song lyrics
    response = requests.get(song['response']['song']['url'], headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    lyrics = soup.find('div', class_='lyrics').get_text()
    lyrics = lyrics.replace('\n', ' ').replace('\r', ' ')
    song_dict['lyrics'] = lyrics
    # print(song_dict)
    return song_dict


@app.route("/results", methods=["GET", "POST"])
def results():
    try:
        print(request.values)
        if request.method == "POST":
            song_data_dict = get_song_details(request.form["song_id"])
            bar_data = build_wordcloud(song_data_dict["lyrics"])
            # print(bar_data)
            build_cloud(song_data_dict["lyrics"])
            return render_template("base.html", song_data=song_data_dict, bar_data=bar_data)
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
