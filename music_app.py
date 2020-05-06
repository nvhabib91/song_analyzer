import requests
import json
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask import abort
from bs4 import BeautifulSoup

app = Flask(__name__)

GENIUS_API_KEY = 'iPRpGvyPEPwHqexfQo75LsL0i2pPQxhkw-P5WStYbdvmUq-PQyf7ppCnT92Z-ZQc'

# class Song(object):
#     def __init__(id, title, artist, url, album, year, image, lyrics ):
#     self.id = id
#     self.title = title
#     self.

# def get_song_details(song_id):
#     base_url = 'https://api.genius.com'
#     genius_base_url = "http://genius.com"
#     headers = {'Authorization': 'Bearer ' + GENIUS_API_KEY}
#     search_url = base_url + '/search'
#     data = {'q': query_string}
#     response = requests.get(search_url, data=data, headers=headers)
#     json = response.json()
#     song_results = []
#     for hit in json['response']['hits']:
#         song_dict = {}
#         song_dict['TitleArtist'] = hit['result']['full_title']
#         song_dict['URL'] = genius_base_url+hit['result']['path']
#         song_results.append(song_dict)


@app.route("/results", methods=["GET", "POST"])
def results():
    try:
        lyrics = ""
        print(request.values)
        if request.method == "POST":
            lyrics_url_form = request.form["song_url"]
            song_title = request.form["q"]
            headers = {'Authorization': 'Bearer ' + GENIUS_API_KEY}
            response = requests.get(lyrics_url_form, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            lyrics = soup.find('div', class_='lyrics').get_text()
            lyrics = lyrics.replace('\n', ' ').replace('\r', ' ')
            return render_template("base.html", pg_title=song_title, page_content=lyrics)
    except:
        return redirect(url_for("index", json_content={'Lyrics Not Found'}, message_type="Error", message_content="Lyrics Not Found"))



@app.route("/_autocomp/<search>")
def autocomplete(search):
    query_string = search
    base_url = 'https://api.genius.com'
    genius_base_url = "http://genius.com"
    headers = {'Authorization': 'Bearer ' + GENIUS_API_KEY}
    search_url = base_url + '/search'
    data = {'q': query_string}
    response = requests.get(search_url, data=data, headers=headers)
    json = response.json()
    song_results = []
    for hit in json['response']['hits']:
        song_dict = {}
        song_dict['TitleArtist'] = hit['result']['full_title']
        song_dict['URL'] = genius_base_url+hit['result']['path']
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
