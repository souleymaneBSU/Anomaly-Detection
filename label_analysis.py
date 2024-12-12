import random
import matplotlib.pyplot as plt

# Function to read the file and process it
def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]  # Read and remove any leading/trailing whitespace
    return lines

# Function to analyze bad lines distribution and find consecutive bad lines
def analyze_bad_lines_distribution(lines, group_size=100):
    bad_indices = [i for i, line in enumerate(lines) if line == 'bad']
    
    # Group bad lines by the group size
    group_bad_count = []
    for i in range(0, len(lines), group_size):
        group = lines[i:i+group_size]
        bad_count = sum(1 for line in group if line == 'bad')
        group_bad_count.append(bad_count)

    # Plot the distribution of bad lines in groups
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(group_bad_count)), group_bad_count, color='red')
    plt.xlabel('Group Number (Size: {})'.format(group_size))
    plt.ylabel('Number of "Bad" Lines')
    plt.title('Distribution of "Bad" Lines Across Groups')
    plt.show()

    # Check for consecutive bad lines and record streaks
    streaks = []
    current_streak = 0
    for i in range(len(lines)):
        if lines[i] == 'bad':
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:  # if the last lines were a streak
        streaks.append(current_streak)

    # Filter streaks of length >= 2 and count occurrences
    streak_lengths = {}
    for streak in streaks:
        if streak >= 2:
            if streak not in streak_lengths:
                streak_lengths[streak] = 0
            streak_lengths[streak] += 1

    # Plot histogram of streak lengths
    plt.figure(figsize=(10, 6))
    plt.bar(streak_lengths.keys(), streak_lengths.values(), color='green')
    plt.xlabel('Length of Streaks of "Bad" Lines')
    plt.ylabel('Frequency')
    plt.title('Histogram of Streak Lengths of "Bad" Lines (Length >= 2)')
    plt.xticks(range(2, max(streak_lengths.keys()) + 1))
    plt.show()

    print("Consecutive 'Bad' Streaks (length >= 2):", streak_lengths)

# Main execution
file_path = 'labels.txt'  # Path to your file
lines = read_file(file_path)



# Call the function to analyze the shuffled lines
analyze_bad_lines_distribution(lines)
