import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import os

# Set global style for seaborn
sns.set_style("whitegrid")

# Helper function to extract raw unicode escape sequences (for emojis)
def extract_raw_unicode(text):
    return [f'\\u{ord(c):04x}' for c in text if ord(c) > 127]

# Clean the text by removing unwanted characters (keeping numbers and alphabets)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep numbers, alphabets, and spaces
    return text

# Load and parse the JSON data
def load_data(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        raw_data = file.read()
        data = json.loads(raw_data)

    messages = []
    for message in data.get("messages", []):
        messages.append({
            'sender': message.get('sender_name', 'Unknown'),
            'timestamp': datetime.fromtimestamp(message.get('timestamp_ms', 0) / 1000),
            'content': message.get('content', ''),
            'reactions': message.get('reactions', [])
        })
    return pd.DataFrame(messages)

# Function to remove 'Meta AI' users
def remove_meta_ai(df):
    return df[df['sender'] != 'Meta AI']

# Filter data by date
def filter_by_date(df, start_date, end_date):
    return df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

# Function to save the plots to Downloads folder with the user's name
def save_plot(filename, user=None):
    # Get the path to the user's Downloads folder
    download_path = os.path.join(os.path.expanduser('~'), 'Downloads')
    
    # If a user name is provided, add it to the filename
    if user:
        filename = f"{user}_{filename}"

    # Check if file already exists and append a number if necessary
    file_path = os.path.join(download_path, filename)
    base_name, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(file_path):
        filename = f"{base_name}_{counter}{ext}"
        file_path = os.path.join(download_path, filename)
        counter += 1
    
    # Save the figure to the Downloads folder
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

# Analysis functions
def avg_messages_per_day(df, user=None):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
    daily_msgs = df.groupby(df['timestamp'].dt.date).size()
    daily_msgs = daily_msgs.reindex(date_range.date, fill_value=0)
    
    plt.figure(figsize=(14, 8))
    plt.plot(daily_msgs.index, daily_msgs.values, marker='o', color='coral', linewidth=2)
    plt.title('Messages per Day', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Messages per Day', fontsize=12)
    plt.xticks(daily_msgs.index[::30], rotation=45)
    
    save_plot('messages_per_day.png', user)

def messages_summary(df, user=None):
    total_messages = len(df)
    total_reactions = sum(len(reactions) for reactions in df['reactions'])
    total_message_length = df['content'].apply(len).sum()
    user_summary = df.groupby('sender').agg(
        total_messages=('content', 'size'),
        total_reactions=('reactions', lambda x: sum(len(reaction) for reaction in x)),
        total_message_length=('content', lambda x: sum(len(msg) for msg in x))
    )
    print("\n=== Total Counts (by User) ===")
    print(user_summary)
    print(f"\nTotal Messages: {total_messages}")
    print(f"Total Reactions: {total_reactions}")
    print(f"Total Message Length: {total_message_length} characters")

def messages_per_user(df, user=None):
    user_counts = df.groupby(['sender', df['timestamp'].dt.to_period('W')]).size().unstack(fill_value=0)
    user_counts = user_counts.T  # Transpose for line plot compatibility
    user_counts.plot(kind='line', marker='o', figsize=(14, 8), linewidth=2)
    plt.title('Messages per User (Week-over-Week)', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Number of Messages', fontsize=12)
    plt.legend(title='User')
    plt.xticks(rotation=45)
    
    save_plot('messages_per_user.png', user)

def avg_message_length(df, user=None):
    df['msg_length'] = df['content'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Group by sender and day to calculate average message length per user per day
    avg_length = df.groupby([df['sender'], df['timestamp'].dt.date])['msg_length'].mean().unstack(fill_value=0)
    
    # Create a complete date range from the first to the last message's date
    full_date_range = pd.date_range(start=df['timestamp'].min().date(), end=df['timestamp'].max().date(), freq='D')
    
    # Reindex the average length dataframe to include all days in the full date range
    avg_length = avg_length.reindex(columns=full_date_range.date, fill_value=0)
    
    # Plot each user's average message length on the same chart
    plt.figure(figsize=(14, 8))
    for user in avg_length.index:
        plt.plot(avg_length.columns, avg_length.loc[user], marker='o', label=user, linewidth=2)

    plt.title('Average Message Length per User (Day-over-Day)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Message Length', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='User')

    save_plot('average_message_length_per_user.png', user)

def most_common_words_per_user(df, top_n=25, user=None):
    user_words = {}
    for user, group in df.groupby('sender'):
        words = Counter()
        for content in group['content'].dropna():
            cleaned_text = clean_text(content.lower())
            words.update(re.findall(r'\b\w+\b', cleaned_text))
        user_words[user] = words.most_common(top_n)

    # Plotting the most common words per user
    for user, words in user_words.items():
        common_words = pd.DataFrame(words, columns=['Word', 'Frequency'])
        sns.barplot(data=common_words, x='Frequency', y='Word', palette="muted")
        plt.title(f'Top {top_n} Most Common Words for {user}', fontsize=16)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Word', fontsize=12)
        
        save_plot(f'{user}_common_words.png', user)

def most_used_emojis_per_user(df, top_n=10, user=None):
    user_emojis = {}
    for user, group in df.groupby('sender'):
        emoji_counter = Counter()
        for content in group['content'].dropna():
            emojis = extract_raw_unicode(content)
            emoji_counter.update(emojis)
        user_emojis[user] = emoji_counter.most_common(top_n)

    # Plotting the most used emojis per user
    for user, emojis in user_emojis.items():
        common_emojis = pd.DataFrame(emojis, columns=['Emoji', 'Frequency'])
        sns.barplot(data=common_emojis, x='Frequency', y='Emoji', palette="Blues_d")
        plt.title(f'Most Used Emojis for {user}', fontsize=16)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Emoji', fontsize=12)
        
        save_plot(f'{user}_most_used_emojis.png', user)

def most_used_reactions_per_user(df, user=None):
    user_reactions = {}
    for user, group in df.groupby('sender'):
        reactions = Counter()
        for reaction_list in group['reactions']:
            if reaction_list:
                for reaction in reaction_list:
                    for char in reaction['reaction']:
                        if ord(char) > 127:
                            reactions.update([f'\\u{ord(char):04x}'])
        user_reactions[user] = reactions.most_common(10)

    # Plotting the most used reactions per user
    for user, reactions in user_reactions.items():
        common_reactions = pd.DataFrame(reactions, columns=['Reaction', 'Frequency'])
        sns.barplot(data=common_reactions, x='Frequency', y='Reaction', palette="viridis")
        plt.title(f'Most Used Reactions for {user}', fontsize=16)
        plt.xlabel('Frequency', fontsize=12)
        plt.ylabel('Reaction', fontsize=12)
        
        save_plot(f'{user}_most_used_reactions.png', user)

# Main function to execute the analysis
def main(file_path, start_date, end_date):
    df = load_data(file_path)
    df = remove_meta_ai(df)  # Remove "Meta AI" user

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = filter_by_date(df, start_date, end_date)

    # Weekly message count per user
    messages_per_user(df)

    # Average message length
    avg_message_length(df)

    # Average messages per day (with explicit day counts)
    avg_messages_per_day(df)

    # Most common words per user (top 25)
    most_common_words_per_user(df, top_n=25)

    # Most used emojis per user (as raw unicode escape sequences)
    most_used_emojis_per_user(df)

    # Most used reactions per user (as raw unicode escape sequences)
    most_used_reactions_per_user(df)

    # Show total counts
    messages_summary(df)
# Run the analysis
file_paths = [r"C:\Users\jishn\Python\inbox\Data\113_8093012077417211\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\aaryatoney_17877965864958819\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\dramaqueens2_7084101538289875\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\gremlinsthathavepublishedabookmanifestation_7730137343773757\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\hetadarji_1026461105471443\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\jankidarji_17847917421061277\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\norazone_17849918514061277\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\norazone_17849918514061277\message_2.json",
              r"C:\Users\jishn\Python\inbox\Data\nrikzi__17842838160082871\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\samaydalal_17843604438269050\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\thethirdwheeliswhatmakesitatricycle_6984097265000622\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\threeidiots_9395937077088375\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\tobi_17848003542061277\message_1.json",
              r"C:\Users\jishn\Python\inbox\Data\vedanshi_17846303412228160\message_1.json"]
start_date = '2023-01-01'
end_date = '2024-12-31'
for filee in file_paths:
    main(filee, start_date, end_date)
