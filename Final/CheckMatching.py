import csv

try:
    with open("KJS_data.csv", "r", encoding='utf-8') as file:
        content = csv.DictReader(file)  # Use DictReader to read the file as a dictionary
        header = content.fieldnames  # Get the header names
        """
        with open('Not_Selected_with_probabilities.csv', 'a', newline='') as file:
            writer_object = csv.writer(file)
            writer_object.writerow(["FirstName", "LastName", "NaiveBayes_Prob", "LogisticRegression_Prob"])  # Replace with actual column names
        """    
        for row in content:
            try:
                if row['Final Selection(750)'] == "Selected":  # Assuming column 152 is named 'SelectionStatus'
                    with open('Selected_with_probabilities.csv', 'a', newline='') as file:
                        writer_object = csv.writer(file)
                        writer_object.writerow([row['Firstname'], row['Last Name'], row['NaiveBayes_Prob'], row['LogisticRegression_Prob']])  # Replace with actual column names
                # print("Added")
            except ValueError:
                # Skip rows with invalid or missing data
                continue

except PermissionError:
    print("Permission denied: Unable to read the file.")
except FileNotFoundError:
    print("File not found: Please check the file path.")
except UnicodeDecodeError:
    print("Error decoding file: Check the file encoding and try again.")
