import pandas as pd  # Import pandas library for data manipulation and analysis
import isodate  # Import isodate library for working with durations
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for data visualization
import matplotlib.ticker as ticker  # Import matplotlib.ticker for custom tick formatting
import seaborn as sns  # Import seaborn for enhanced data visualization
sns.set(style="darkgrid", color_codes=True)  # Set the seaborn style and color codes

# Import the required libraries for the Google API
from googleapiclient.discovery import build

# Import the required libraries for NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Import the required libraries for saving figures to PDF
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Set the YouTube API key, service name, and version
api_key = 'AIzaSyCDEEqn35XQVXRvAx9hSeLLThjTDBeB78A'  # Replace with your actual API key
api_service_name = "youtube"
api_version = "v3"

# Build the YouTube API service using the API key
youtube = build(api_service_name, api_version, developerKey=api_key)


def get_video_ids(youtube, playlist_id):
    """
    Retrieves the video IDs of all videos in a YouTube playlist.

    Params:
        youtube: The build object from googleapiclient.discovery.
        playlist_id: The ID of the YouTube playlist.

    Returns:
        List of video IDs in the playlist.
    """
    # Create a request to retrieve the contentDetails (video IDs) of videos in a playlist
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50)
    # Execute the request and obtain the response
    response = request.execute()

    # Initialize an empty list to store the video IDs
    video_ids = []

    # Iterate over the items in the response and extract the video IDs
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    # Check if there are more pages (more videos in the playlist)
    next_page_token = response.get('nextPageToken')
    more_pages = True

    # Continue retrieving video IDs from subsequent pages if available
    while more_pages:
        # If there is no next page token, set more_pages to False to exit the loop
        if next_page_token is None:
            more_pages = False
        else:
            # Create a new request for the next page using the page token
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token)
            # Execute the request for the next page and obtain the response
            response = request.execute()

            # Iterate over the items in the response and append the video IDs
            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])

            # Update the next page token for the next iteration
            next_page_token = response.get('nextPageToken')

    # Return the list of video IDs
    return video_ids

# choose a playlist and paste the playlist id here
playlist_id = "PL0vfts4VzfNjQOM9VClyL5R0LeuTxlAR3"
# get the video ids
video_ids = get_video_ids(youtube, playlist_id)


def get_video_details(youtube, video_ids):
    """
    Get video statistics of all videos with given IDs

    Params:
        youtube: The build object from googleapiclient.discovery.
        video_ids: List of video IDs.

    Returns:
        Dataframe with statistics of videos, i.e.:
        'channelTitle', 'title', 'description', 'tags', 'publishedAt'
        'viewCount', 'likeCount', 'favoriteCount', 'commentCount'
        'duration', 'definition', 'caption'
    """
    # Initialize an empty list to store the video information
    all_video_info = []

    # Iterate over the video IDs in batches of 50 (YouTube API's maximum limit for video IDs per request)
    for i in range(0, len(video_ids), 50):
        # Create a request to retrieve video details including snippet, contentDetails, and statistics
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i + 50])
        )
        # Execute the request and obtain the response
        response = request.execute()

        # Iterate over the videos in the response
        for video in response['items']:
            # Define a dictionary to store the specific statistics and information for each video
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                             }
            video_info = {}
            video_info['video_id'] = video['id']

            # Iterate over the keys in the stats_to_keep dictionary
            for k in stats_to_keep.keys():
                # Iterate over the values (specific fields) corresponding to each key
                for v in stats_to_keep[k]:
                    try:
                        # Try to retrieve the specific field from the video object
                        video_info[v] = video[k][v]
                    except:
                        # If the field is not present, set the value in video_info to None
                        video_info[v] = None

            # Append the video_info dictionary to the all_video_info list
            all_video_info.append(video_info)

    # Convert the list of video information to a pandas DataFrame and return it
    return pd.DataFrame(all_video_info)



# get the video details
video_df = get_video_details(youtube, video_ids)


def get_comments_in_videos(youtube, video_ids):
    """
    Get top level comments as text from all videos with given IDs (only the first 10 comments due to quote limit of Youtube API)

    Params:
        youtube: The build object from googleapiclient.discovery.
        video_ids: List of video IDs.

    Returns:
        Dataframe with video IDs and associated top-level comments in text.
    """
    # Initialize an empty list to store the comments
    all_comments = []

    # Iterate over the video IDs
    for video_id in video_ids:
        try:
            # Create a request to retrieve comment threads for the video
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            )
            # Execute the request and obtain the response
            response = request.execute()

            # Extract the text of top-level comments from the response
            comments_in_video = [comment['snippet']['topLevelComment']['snippet']['textOriginal'] for comment in
                                 response['items']]
            # Create a dictionary to store the video ID and associated comments
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

            # Append the comments_in_video_info dictionary to the all_comments list
            all_comments.append(comments_in_video_info)

        except:
            # Handle the case when an error occurs (most likely due to comments being disabled on the video)
            print('Could not get comments for video ' + video_id)

    # Convert the list of comments information to a pandas DataFrame and return it
    return pd.DataFrame(all_comments)

# get the comments
comments_df = get_comments_in_videos(youtube, video_ids)



# Ensure 'publishedAt' is in datetime format
video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'])

# Convert 'duration' to duration in seconds
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())

# Convert 'commentCount' and 'likeCount' to numeric
video_df['commentCount'] = pd.to_numeric(video_df['commentCount'])
video_df['likeCount'] = pd.to_numeric(video_df['likeCount'])

# Count the number of tags in 'tags' column
video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))

# Convert 'viewCount' column to numeric
video_df['viewCount'] = pd.to_numeric(video_df['viewCount'])

# Convert 'publishedAt' to datetime format
video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'])

# Extract the day name from 'publishedAt'
video_df['publishedDay'] = video_df['publishedAt'].dt.day_name()








# Create a figure with two subplots
fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
plt.subplots_adjust(bottom=0.557, top=0.964, left=0.125, right=0.9)

# Create the first bar plot
sns.barplot(x='title', y='viewCount', data=video_df.sort_values("viewCount", ascending=False)[0:9], ax=axes[0])
axes[0].set_title('Top 10 videos by view count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

# Format the y-axis tick labels to show values in K (thousands) or M (millions)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K' if x > 1000 and x < 1000000 else '{:,.0f}'.format(x / 1000000) + 'M' if x >= 1000000 else x))

# Create the second bar plot
sns.barplot(x='title', y='viewCount', data=video_df.sort_values('viewCount', ascending=True)[0:9], ax=axes[1])
axes[1].set_title('Bottom 10 videos by view count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

# Format the y-axis tick labels to show values in K (thousands) or M (millions)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K' if x > 1000 and x < 1000000 else '{:,.0f}'.format(x / 1000000) + 'M' if x >= 1000000 else x))


def human_format(num):
    """
    Converts a large number into a human-readable format with magnitude suffix (K, M, etc.)

    Params:
        num: The number to be formatted.

    Returns:
        The formatted number as a string with magnitude suffix.
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# Create a new figure with two subplots
fig2, ax = plt.subplots(1, 2, figsize=(20, 10))

# First scatter plot
sns.scatterplot(data=video_df, x='commentCount', y='viewCount', ax=ax[0])
ax[0].set_title('View count vs Comment count')
ax[0].set_xlabel('Comment Count')
ax[0].set_ylabel('View Count')

# Format the x-axis and y-axis tick labels using the human_format function
ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))

# Set the x-axis tick labels for better readability
ax[0].set_xticklabels(ax[0].get_xticks())

# Second scatter plot
sns.scatterplot(data=video_df, x='likeCount', y='viewCount', ax=ax[1])
ax[1].set_title('View count vs Like count')
ax[1].set_xlabel('Like Count')
ax[1].set_ylabel('View Count')

# Format the x-axis and y-axis tick labels using the human_format function
ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))

# Set the x-axis tick labels for better readability
ax[1].set_xticklabels(ax[1].get_xticks())



# Create a new figure
fig4, ax = plt.subplots(figsize=(20,10))

# Create a histogram of video durations
sns.histplot(data=video_df, x='durationSecs', bins=30)
ax.set_title('Distribution of Video Duration')
ax.set_xlabel('Duration (seconds)')

weekday_counts = video_df['publishedDay'].value_counts().to_dict()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a DataFrame with weekday counts
day_df = pd.DataFrame(list(weekday_counts.items()), columns=['Day', 'Count'])
day_df['Day'] = pd.Categorical(day_df['Day'], categories=weekdays, ordered=True)
day_df = day_df.sort_values('Day')

fig5, ax = plt.subplots(figsize=(20, 10))

# Create a bar plot of the number of videos published on each day of the week
sns.barplot(x='Day', y='Count', data=day_df, ax=ax, palette='viridis')
ax.set_title('Number of Videos Published on Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Videos')


# Download the required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set up Russian stopwords
stop_words = set(stopwords.words('english'))

# Apply stopwords removal to title column
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in word_tokenize(str(x)) if item.lower() not in stop_words])
comments_df['comments_no_stopwords'] = comments_df['comments'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
all_words_2 = list([a for b in comments_df['comments_no_stopwords'].tolist() for a in b])
all_words_str_2 = ' '.join(all_words_2)

# Flatten the list of words
all_words = [word for sublist in video_df['title_no_stopwords'] for word in sublist]
all_words_str = ' '.join(all_words)


def plot_cloud(wordcloud):
    """
    Plot and display the word cloud.

    Params:
        wordcloud: The WordCloud object to be plotted.
    """
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud)
    plt.axis("off")


# Generate and display the first word cloud
wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)

# Generate and display the second word cloud
wordcloud_2 = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                        colormap='viridis', collocations=False).generate(all_words_str_2)
plot_cloud(wordcloud_2)

wordcloud.to_file('key_frases_of_title.png')  # Save the word cloud as an image
img = mpimg.imread('key_frases_of_title.png')  # Read the saved image
fig3, ax = plt.subplots(figsize=(20,10))
ax.imshow(img)
ax.axis('off')

wordcloud_2.to_file('key_frases_of_comments.png')  # Save the word cloud as an image
img = mpimg.imread('key_frases_of_comments.png')  # Read the saved image
fig6, ax = plt.subplots(figsize=(20,10))
ax.imshow(img)
ax.axis('off')

with PdfPages('all_stats.pdf') as pdf:
    fig1.suptitle('All stats')  # Add a title
    fig1.savefig(pdf, format='pdf')  # Save figure 1 to the PDF
    fig2.savefig(pdf, format='pdf')  # Save figure 2 to the PDF
    fig4.savefig(pdf, format='pdf')  # Save figure 4 to the PDF
    fig5.savefig(pdf, format='pdf')  # Save figure 5 to the PDF
    fig3.suptitle('Key phrases in title of videos')  # Add a title to figure 3
    fig3.savefig(pdf, format='pdf')  # Save figure 3 to the PDF
    fig6.suptitle('Key phrases in comment below videos')  # Add a title to figure 6
    fig6.savefig(pdf, format='pdf')  # Save figure 6 to the PDF



