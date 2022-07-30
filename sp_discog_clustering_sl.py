# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:06:40 2022

@author: Steve

Cluster an artist's discography based on audio features obtained
from Spotify.
"""

import numpy as np
import pandas as pd
import pprint

import spotipy
import spotipy.cache_handler as ch
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

import os
from bs4 import BeautifulSoup
import requests
from requests.exceptions import ReadTimeout

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

"""
Set up connection to Spotify
"""

scope = 'user-library-read'

# Pick which credentials to use to prevent timeout problems
use_credential_set = 1

if use_credential_set == 1:
    os.environ['SPOTIPY_CLIENT_ID'] = client_id
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
    os.environ['SPOTIPY_REDIRECT_URI'] = redirect_url
elif use_credential_set == 2:
    os.environ['SPOTIPY_CLIENT_ID'] = client_id2
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret2
    os.environ['SPOTIPY_REDIRECT_URI'] = redirect_url2
elif use_credential_set == 3:
    os.environ['SPOTIPY_CLIENT_ID'] = client_id3
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret3
    os.environ['SPOTIPY_REDIRECT_URI'] = redirect_url3

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

"""
Define functions that will be used to obtain artist,
album, and track information
"""

# Get track IDs for all tracks on specified album.
# Note that this function searches by album name and
# selects the first album returned by query.
def get_album_track_ids(album_name, spy):
    album_results = spy.search(q='album:'+album_name, type='album')
    try:
        album_id = album_results['albums']['items'][0]['id']
        album = spy.album(album_id)

        album_track_ids = []

        for i in range(len(album['tracks']['items'])):
            album_track_ids.append(album['tracks']['items'][i]['id'])

        return album_track_ids
    except:
        return None

def get_album_track_features(album_name, spy):
    track_ids = get_album_track_ids(album_name, spy)
    if track_ids:
        track_features = [spy.audio_features(x)[0] for x in track_ids]
        for features in track_features:
            try:
                features['name'] = spy.track(features['id'])['name']
            except:
                features['name'] = None
            try:
                features['album_name'] = album_name
            except:
                features['album_name'] = None
            try:
                features['artist'] = spy.track(features['id'])['artists'][0]['name']
            except:
                features['artist'] = None
        return track_features
    else:
        return None

def get_album_track_features_df(album_name, spy):
    features_list = get_album_track_features(album_name, spy)

    pitch_class = {
        0: 'C',
        1: 'C#/Db',
        2: 'D',
        3: 'D#/Eb',
        4: 'E',
        5: 'F',
        6: 'F#/Gb',
        7: 'G',
        8: 'G#/Ab',
        9: 'A',
        10: 'A#/Bb',
        11: 'B'
    }

    if features_list:
        df = pd.DataFrame(features_list)

        df['key_letter'] = [pitch_class[x] for x in df['key']]

        return df

def get_artist_discog(artist_name):
    artist_id = sp.search(artist_name, type='artist')['artists']['items'][0]['id']
    albums = sp.artist_albums(artist_id, limit=50)
    albums_list = []
    for album in albums['items']:
        album_entry = {
            'id': album['id'],
            'album_name': album['name'],
            'release_date': album['release_date'],
            'image_url': album['images'][0]['url']
        }
        albums_list.append(album_entry)
    album_df = pd.DataFrame(albums_list)
    album_df.drop_duplicates('release_date', inplace=True)
    return album_df

def get_artist_tracks(artist_name):
    discog = get_artist_discog(artist_name)
    album_df_list = []
    for album_name in discog['album_name']:
        album_df = get_album_track_features_df(album_name, sp)
        album_df_list.append(album_df)
    songs_df = pd.concat(album_df_list)
    return songs_df

"""
Use seaborn to create a heatmap of correlation values
between different audio features provided by Spotify
"""
def corr_plot_from_songs_df(songs_df, by_album=False, color_map='mako'):
#     corr_matrix = songs_df.groupby(by='album_name').mean().corr().round(2)
    if by_album:
        corr_matrix = songs_df.groupby(by='album_name').mean().corr().round(2)
    else:
        corr_matrix = songs_df.corr().round(2)
    sns.set(rc = {'figure.figsize':(10,10)})
    sns.heatmap(corr_matrix, annot=True, cmap=color_map)
    plt.title(songs_df['artist'].iloc[0])

"""
Define a function to take in an artist's discography as a
pandas DataFrame and cluster the albums based on audio
features provided by Spotify. Note that this works better
when the artist has a large number of albums, and may not
work at all if the artist has a very small number of albums.
"""
def cluster_discog_from_df_kmeans(artist_discog, min_k=3, max_k=10, print_result=False):
    X = np.array(artist_discog.groupby('album_name').mean())#.drop(['album_name', 'artist', 'key_letter'], axis=1))
    album_titles = np.array(artist_discog.groupby('album_name').mean().index)

    sil_scores = []

    print('Optimizing clustering by number of clusters (k)...')

    for k in range(min_k, max_k+1):
        print('Testing clustering with k = ' + str(k))
        kmeans = KMeans(n_clusters=k).fit(X)
        labels = kmeans.labels_
        sil_scores.append([k, silhouette_score(X, labels, metric='euclidean')])

    sil_scores_df = pd.DataFrame(np.array(sil_scores))
    print(sil_scores_df)
    optimal_k = int(sil_scores_df[sil_scores_df[1]==max(sil_scores_df[1])].to_numpy()[0][0])

    print('Clustering optimized.')
    print('Optimal number of clusters = ' + str(optimal_k) + '\n')

    optimal_clustering = KMeans(n_clusters=optimal_k).fit(X)
    clustered_df = pd.DataFrame(np.transpose(np.array([album_titles, optimal_clustering.labels_])))

    clustered_albums_str = ''

    for k in range(optimal_k):
        # print('<| k = ' + str(k+1) + ' |>')
        clustered_albums_str += ('<| k = ' + str(k+1) + ' |>\n')
        for album in clustered_df[clustered_df[1]==k][0]:
            # print(album)
            clustered_albums_str += (album + '\n')
        # print('-'*36)
        clustered_albums_str += ('-'*36 + '\n')

    if print_result == True:
        print(clustered_albums_str)

    return clustered_albums_str

def cluster_artist_discog_from_name(artist_name, min_k=3, max_k=10, print_results=False):
    print('Obtaining artist discography information...')
    artist_discog = get_artist_tracks(artist_name)
    print('Obtained artist discography information.')
    clustered_albums_str = cluster_discog_from_df_kmeans(artist_discog, min_k, max_k, print_results)
    return clustered_albums_str
