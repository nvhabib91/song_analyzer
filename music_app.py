import requests
import sqlalchemy
# from flask_sqlalchemy import SQLAlchemy
import json
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
from flask import Flask, jsonify, render_template, request, redirect, url_for
from flask import abort
from sqlalchemy import or_
from sqlalchemy import and_
from sqlalchemy import any_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from bs4 import BeautifulSoup

Base = declarative_base()


engine = create_engine("sqlite:///static/data/musicdb.sqlite")
conn = engine.connect()

# Base = automap_base()
# Base.prepare(engine, reflect=True)
# print(Base.classes.keys())
# Table NEEDS to have a primary key to be able to be automapped
# Songs = Base.classes.songs_vt

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///static/data/musicdb.sqlite"
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)

class Songs(Base):
    __tablename__ = 'songs_vt'
    SONG_TITLE_ARTIST = Column(String)
    SONG_URL = Column(String, primary_key=True)
    RANK = Column(Integer)

#     def __repr__(self):
#         return '<Song %r>' % (self.SONG_URL)

# Create database classes
# @app.before_first_request
# def setup():
#     db.create_all()


@app.route("/results", methods=["GET", "POST"])
def results():
    try:
        lyrics = ""
        # print(request.values)
        if request.method == "POST":
            lyrics_url_form = request.form["song_url"]
            song_title = request.form["q"]
            lyrics_url = lyrics_url_form
            response = requests.get(lyrics_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            lyrics = soup.find('div', class_=None).text
            # lyrics = lyrics.replace('\n', ' ').replace('\r', ' ')
            return render_template("base.html", pg_title=song_title, page_content=lyrics)
    except:
        # return redirect(url_for("/", pg_message="Lyrics Not Found"))
        return redirect(url_for("index", json_content={'Lyrics Not Found'}, message_type="Error", message_content="Lyrics Not Found"))



@app.route("/_autocomp/<search>")
def autocomplete(search):
    query = search
    q_clean = ' '.join(query.strip().split())
    q_split = q_clean.split(' ')
    q_temp = [a+'*' if (idx == 0 & len(q_split) != 1) else 'OR '+a+'*' for idx, a in enumerate(q_split)]
    q_final = ' '.join(q_temp)
    # print(f"{q_final}")
    
    session = Session(bind=engine)
    results = session.query(Songs.SONG_TITLE_ARTIST, Songs.SONG_URL).filter(
        Songs.SONG_TITLE_ARTIST.match(q_final)).order_by(Songs.RANK).limit(10).all()
    session.close()

    # print(results) 
    song_results = []
    # format into JSON
    for SONG_TITLE_ARTIST, SONG_URL in results:
        song_dict = {}
        song_dict['TitleArtist'] = SONG_TITLE_ARTIST.title()
        # song_dict['Artist'] = ARTIST_NAME
        song_dict['URL'] = SONG_URL
        song_results.append(song_dict)
    # song_results = []
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
    # print(json_content, message_type, message_content)
    return render_template('index.html', json_content=json_content, message_type=message_type, message_content=message_content)


if __name__ == "__main__":
    app.run(debug=True)
