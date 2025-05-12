import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set styles
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# Load the data
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')

# ========== 1. Matches per Season ==========
season_matches = matches['season'].value_counts().sort_index()
plt.plot(season_matches.index, season_matches.values, marker='o', color='blue')
plt.title('Matches per Season')
plt.xlabel('Season')
plt.ylabel('Number of Matches')
plt.grid(True)
plt.show()

# ========== 2. Most Successful Teams ==========
team_wins = matches['winner'].value_counts()
plt.plot(team_wins.index, team_wins.values, marker='o', color='green')
plt.title('Most Winning Teams')
plt.xlabel('Teams')
plt.ylabel('Wins')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# ========== 3. Top 10 Run Scorers ==========
top_batsmen = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)
plt.plot(top_batsmen.index, top_batsmen.values, marker='o', color='purple')
plt.title('Top 10 Run Scorers')
plt.xlabel('Batsman')
plt.ylabel('Total Runs')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ========== 4. Top 10 Wicket Takers ==========
wickets = deliveries[deliveries['dismissal_kind'].notnull()]
top_bowlers = wickets['bowler'].value_counts().head(10)
plt.plot(top_bowlers.index, top_bowlers.values, marker='o', color='teal')
plt.title('Top 10 Wicket Takers')
plt.xlabel('Bowler')
plt.ylabel('Wickets')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ========== 5. Player 50+ Performances ==========
batsman_scores = deliveries.groupby(['match_id', 'batter'])['batsman_runs'].sum().reset_index()
fifties = batsman_scores[batsman_scores['batsman_runs'] >= 50]
top_fifty_getters = fifties['batter'].value_counts().head(5)
plt.plot(top_fifty_getters.index, top_fifty_getters.values, marker='o', color='orange')
plt.title('Top 5 Players with Most 50+ Scores')
plt.xlabel('Batsman')
plt.ylabel('No. of 50+ scores')
plt.grid(True)
plt.show()

# ========== 6. Team-wise Analysis ==========
team_name = "Mumbai Indians"

team_matches = matches[(matches['team1'] == team_name) | (matches['team2'] == team_name)]
total_matches = team_matches.shape[0]
wins = matches[matches['winner'] == team_name].shape[0]
losses = total_matches - wins
win_pct = (wins / total_matches) * 100 if total_matches != 0 else 0

print(f"\n📊 {team_name} Analysis:")
print(f"Total Matches: {total_matches}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Win Percentage: {win_pct:.2f}%")

# ========== 7. Wins per Season ==========
team_season_wins = matches[matches['winner'] == team_name]['season'].value_counts().sort_index()
team_season_total = team_matches['season'].value_counts().sort_index()
win_percentage = (team_season_wins / team_season_total * 100).fillna(0)

plt.plot(win_percentage.index, win_percentage.values, marker='o', color='blue')
plt.title(f'{team_name} - Win % by Season')
plt.xlabel('Season')
plt.ylabel('Win Percentage')
plt.grid(True)
plt.show()

# ========== 8. Top Venues for the Team ==========
venue_stats = matches[matches['winner'] == team_name]['venue'].value_counts().head(5)
plt.plot(venue_stats.index, venue_stats.values, marker='o', color='green')
plt.title(f'Top 5 Venues where {team_name} Wins Most')
plt.xlabel('Venue')
plt.ylabel('Wins')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# ========== 9. Bat First vs Chase ==========
bat_first = matches[(matches['toss_winner'] == team_name) & (matches['toss_decision'] == 'bat')]
field_first = matches[(matches['toss_winner'] == team_name) & (matches['toss_decision'] == 'field')]
bat_first_wins = bat_first[bat_first['winner'] == team_name].shape[0]
field_first_wins = field_first[field_first['winner'] == team_name].shape[0]

plt.plot(['Bat First', 'Chase'], [bat_first_wins, field_first_wins], marker='o', color='brown')
plt.title(f'{team_name} Wins: Batting First vs Chasing')
plt.ylabel('Number of Wins')
plt.grid(True)
plt.show()

# ========== 10. Match Prediction (Visual) ==========
team1 = "Mumbai Indians"
team2 = "Chennai Super Kings"

head_to_head = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                       ((matches['team1'] == team2) & (matches['team2'] == team1))]

wins_team1 = head_to_head[head_to_head['winner'] == team1].shape[0]
wins_team2 = head_to_head[head_to_head['winner'] == team2].shape[0]

teams = [team1, team2]
wins = [wins_team1, wins_team2]

plt.plot(teams, wins, marker='o', color='black')
# Prediction line
plt.plot(teams, wins, 'r--', label='Prediction Line')
plt.title(f'{team1} vs {team2} - Head to Head')
plt.ylabel('Wins')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n🔮 Prediction: {team1} Wins = {wins_team1}, {team2} Wins = {wins_team2}")
if wins_team1 > wins_team2:
    print(f"Prediction: {team1} might win based on history!")
elif wins_team2 > wins_team1:
    print(f"Prediction: {team2} might win based on history!")
else:
    print("Prediction: It's a close match!")


