import pandas as pd

file_path = "/Users/vishchaudhary/rl-repo/results/" + "2024-11-11_23-37-33-HPT/" + "hpt_results.csv"
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_colwidth', None)  # Display full column width
def main():
    df = pd.read_csv(file_path)
    df.sort_values(by=['final_fidelity'], ascending=False, inplace=True)
    print(df[0:1])
    print(df[1:2])

if __name__ == "__main__":
    main()