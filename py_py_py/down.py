from pytube import Playlist

playlist = Playlist('https://www.youtube.com/playlist?list=PL8yOO2xYRcZt_P1ah4p2DJEOyEOVFJ3Bg')
print('Number of videos in playlist: %s' % len(playlist.video_urls))


for video in playlist.videos:
    video.streams.first().download()
