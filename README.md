# YouTube Video Analytics

This project provides a set of tools to analyze YouTube video data using the YouTube Data API. It retrieves video statistics, comments, and generates visualizations for further analysis.

## Prerequisites

Before running the code, make sure you have the following:

- Python 3.7 or higher installed
- Necessary Python libraries (listed in `requirements.txt`)

## Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/Vladyslav30/Youtube_playlist_project.git
   ```
2. Navigate to the project directory

3.(Optional) Create and activate a virtual environment (recommended)

  On Windows:
  ``` shell
python -m venv venv
.\venv\Scripts\activate
  ```
On macOS and Linux:
``` shell
python3 -m venv venv
source venv/bin/activate
```

4.Install the required Python libraries
``` shell
pip install -r requirements.txt
```

## Usage
1.Run the youtube_playlist_project.py script to retrieve video data and generate analytics
``` shell
python youtube_playlist_project.py
```
2.Open file "all_stats.pdf" and enjoy statisticks that you see

3.You could change playlist_id in 86 string but you must use only playlist id

4.Again run file
``` shell
python youtube_playlist_project.py
```
5.And you will see new statisticks about playlist that you want.



