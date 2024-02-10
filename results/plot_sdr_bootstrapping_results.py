import matplotlib.pyplot as plt
import re
import os

dir_paths = [
    ...
    ]

for directory_path in dir_paths:
    exp_name = directory_path.split('/')[-1]

    # Create empty lists to store the data we want to plot
    x_values = []
    y_values_pass = []
    y_values_repaired = []

    # Set the path to your directory of txt files
    plain_bootstrapping = "plain" in directory_path

    # Use regular expressions to match the filenames we want to read
    if plain_bootstrapping:
        filename_pattern = r'valid_examples_(\d+).txt'
    else:
        filename_pattern = r'repaired_valid_examples_(\d+).txt'

    print(os.listdir(directory_path))

    # Loop over all files in the directory
    for filename in sorted(os.listdir(directory_path)):
        # Check if the filename matches our pattern
        match = re.match(filename_pattern, filename)
        if match:
            # Extract the number from the filename and add it to the x-values list
            x_values.append(int(match.group(1)))

            # Read the first line of the file and extract the pass and repaired percentages
            with open(os.path.join(directory_path, filename)) as file:
                first_line = file.readline().strip()

                if plain_bootstrapping:
                    pass_percent = float(re.search(r'Pass at 1: (\d+\.\d+)', first_line).group(1))
                else:
                    pass_percent = float(re.search(r'Pass at 1: (\d+\.\d+)%', first_line).group(1))
                    repaired_percent = float(re.search(r'pass with repairing: (\d+\.\d+)%', first_line).group(1))

                # Add the percentages to the y-values lists
                y_values_pass.append(pass_percent)

                if not plain_bootstrapping:
                    y_values_repaired.append(repaired_percent)


    if plain_bootstrapping:
        # Sort the values based on the x-values
        x_values, y_values_pass = zip(*sorted(zip(x_values, y_values_pass)))
    else:
        # Sort the values based on the x-values
        x_values, y_values_pass, y_values_repaired = zip(*sorted(zip(x_values, y_values_pass, y_values_repaired)))

    # Create a plot with one or two lines, depending if we're using repairing or not
    plt.plot(x_values, y_values_pass, label=f'{exp_name} Pass@1')

    if not plain_bootstrapping:
        plt.plot(x_values, y_values_repaired, label=f'{exp_name} Pass@1 with Repairing@1')

    # Add a legend and axis labels
    plt.legend()
    plt.xlabel('Bootstrapping steps')
    plt.ylabel('Pass percentage')

    # Add a dot for every value
    plt.scatter(x_values, y_values_pass)

    if not plain_bootstrapping:
        plt.scatter(x_values, y_values_repaired)

    # Set the y-axis limits
    plt.ylim(0, 30)

    # Give the plot a title which is the same as the directory name
    # plt.title(directory_path.split('/')[-1])

    print("Experiment name:", exp_name)
    # Print max score and the corresponding bootstrapping step
    print(f"Max score: {max(y_values_pass)} at step {y_values_pass.index(max(y_values_pass))}")

    # If we're using repairing, also print the max score and the corresponding bootstrapping step for repairing
    if not plain_bootstrapping:
        print(max(y_values_repaired))
        print(x_values[y_values_repaired.index(max(y_values_repaired))])
    print()

# Give title


# Show the plot
plt.show()