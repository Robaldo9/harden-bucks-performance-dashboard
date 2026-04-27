from nba_api.stats.static import players, teams


def main():
    harden_matches = players.find_players_by_full_name("James Harden")
    bucks_matches = teams.find_teams_by_full_name("Milwaukee Bucks")
    blazers_matches = teams.find_teams_by_full_name("Portland Trail Blazers")

    print("")
    print("NBA API TEST")
    print("------------")
    print("James Harden:")
    print(harden_matches)
    print("")
    print("Milwaukee Bucks:")
    print(bucks_matches)
    print("")
    print("Portland Trail Blazers:")
    print(blazers_matches)
    print("")

    if not harden_matches:
        raise ValueError("James Harden not found.")

    if not bucks_matches:
        raise ValueError("Milwaukee Bucks not found.")

    if not blazers_matches:
        raise ValueError("Portland Trail Blazers not found.")

    print("Test passed.")
    print("")


if __name__ == "__main__":
    main()