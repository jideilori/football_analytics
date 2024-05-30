# football_analytics

In this proof of concept, I utilized YOLO and supervision to detect and track players. I recorded the players' foot positions in a CSV file for further analysis. Since the video is a TV broadcast, no perspective transformation was applied, which means an accurate heatmap for each team cannot be created. Below are the results.

##

[CSV file](https://drive.google.com/file/d/1-3he4v0T94aJW6abwwsb8LSOE6QOhUg4/view?usp=sharing)  

|csv column name      | definition                          |
|---------------------|-------------------------------------|
| frame_count         | frame number                        |
| id                  | tracking_id                         |
| x                   | foot position of player - X         |
| y                   | foot position of player - Y         |
| color               | team color                          |


##
<video width="640" height="480" controls>
  <source src="output.mp4" type="video/mp4">
</video>

[Watch the full video on Google Drive](https://drive.google.com/file/d/1rraIt3mPzbKbcUZY_n2Z8vp3P7VXPOso/view?usp=sharing)

