# Course Scheduling Genetic Algorithm

## Overview

This project implements a genetic algorithm to schedule theory and lab courses effectively. The algorithm is designed to create a timetable that satisfies various constraints such as avoiding course overlaps, ensuring that instructors are not double-booked, and managing the number of courses assigned to each section and instructor.

## Features

- **Genetic Algorithm**: Utilizes genetic algorithm techniques including selection, crossover, and mutation to optimize the timetable.
- **Binary Encoding**: Courses, sections, and instructors are encoded in binary to facilitate genetic operations.
- **Fitness Function**: Evaluates timetables based on constraints and penalties to determine the quality of solutions.
- **Data Integration**: Reads course and timetable data from Excel files stored in Google Drive.

## Getting Started

### Prerequisites

- Python 3.x
- Google Colab (for running the notebook and accessing Google Drive)
- Required Python libraries: `pandas`, `numpy`

### Installation

1. **Mount Google Drive**: Ensure you have access to your Google Drive and mount it in the Google Colab environment:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Prepare Excel Files**: Place the following files in your Google Drive:
    - `courses.xlsx`: Contains theory and lab courses information.
    - `lab_courses.xlsx`: Contains lab courses information.
    - `timetable.xlsx`: Contains the timetable for theory courses.
    - `timetable_lab.xlsx`: Contains the timetable for lab courses.

3. **Install Dependencies**:

    ```sh
    pip install pandas numpy
    ```

## Code Description

### Import Libraries and Read Files

```python
import pandas as pd
import numpy as np
import random
from numpy.random import choice

df_theory = pd.read_excel("/content/drive/My Drive/courses.xlsx")
df_lab = pd.read_excel("/content/drive/My Drive/lab_courses.xlsx")
timetable_theory = pd.read_excel("/content/drive/My Drive/timetable.xlsx")
timetable_lab = pd.read_excel("/content/drive/My Drive/timetable_lab.xlsx")
```

### Data Preparation and Encoding
```python
rooms_theory = timetable_theory['Room'].values.tolist()[:30]
timeslots_theory = timetable_theory.columns.values.tolist()[1:]
rooms_lab = timetable_lab['Lab'].values.tolist()[:16]
timeslots_lab = timetable_lab.columns.values.tolist()[1:]

df_combined = pd.concat([df_theory, df_lab], ignore_index=True)
timetable_combined = pd.concat([timetable_theory, timetable_lab], axis=1)

df_theory['Type'] = 'Theory'
df_lab['Type'] = 'Lab'
df_combined = pd.concat([df_theory, df_lab], ignore_index=True)

courses = df_combined['Course'].unique()
sections = df_combined['Section'].unique()
instructors = df_combined['Course Instructor'].unique()

course_bits = len(format(len(courses), 'b'))
section_bits = len(format(len(sections), 'b'))
instructor_bits = len(format(len(instructors), 'b'))

course_mapping = {course: format(index, f'0{course_bits}b') for index, course in enumerate(courses)}
section_mapping = {section: format(index, f'0{section_bits}b') for index, section in enumerate(sections)}
instructor_mapping = {instructor: format(index, f'0{instructor_bits}b') for index, instructor in enumerate(instructors)}

reverse_course_mapping = {v: k for k, v in course_mapping.items()}
reverse_section_mapping = {v: k for k, v in section_mapping.items()}
reverse_instructor_mapping = {v: k for k, v in instructor_mapping.items()}

theory_chromosomes = np.array([
    [course_mapping[row['Course']], section_mapping[row['Section']], instructor_mapping[row['Course Instructor']]]
    for _, row in df_combined[df_combined['Type'] == 'Theory'].iterrows()
])

lab_chromosomes = np.array([
    [course_mapping[row['Course']], section_mapping[row['Section']], instructor_mapping[row['Course Instructor']]]
    for _, row in df_combined[df_combined['Type'] == 'Lab'].iterrows()
])

binary_string_mapping = {
    'course': {v: k for k, v in course_mapping.items()},
    'section': {v: k for k, v in section_mapping.items()},
    'instructor': {v: k for k, v in instructor_mapping.items()}
}

teacher_course_count = {teacher: 0 for teacher in instructor_mapping.values()}
section_course_count = {section: 0 for section in section_mapping.values()}
```

### Fitness Function
```python
def fitness(timetable):
    score = 0

    # If a column doesn't have a section name clash, +40 points
    for x in timetable.T[1]:
        if len(np.unique(x)) == len(x):
            score += 40

    # If a column doesn't have a teacher name clash, +40 points
    for x in timetable.T[2]:
        if len(np.unique(x)) == len(x):
            score += 40

    # Check for clashes of classes at the same time
    if timetable.shape[1] == len(timeslots_theory):
        timeslots = timeslots_theory
    else:
        timeslots = timeslots_lab

    for timeslot_index in range(len(timeslots)):
        section_teacher_pairs = timetable[:, timeslot_index, 1:3]  # Extract section and teacher for the current timeslot
        unique_pairs = np.unique(section_teacher_pairs, axis=0)  # Get unique pairs of section and teacher
        if len(unique_pairs) != len(section_teacher_pairs):  # If there are duplicate pairs, penalize
            score -= 100  # Penalize

    # Check if a professor is assigned more than 3 courses
    for teacher, count in teacher_course_count.items():
        if count > 3:
            score -= (count - 3) * 100  # Penalize

    # Check if no section has more than 5 courses
    for section, count in section_course_count.items():
        if count > 5:
            score -= (count - 5) * 100  # Penalize

    return score
```

### Mutation Function
```python
def mutation(child, probability):
    if random.random() < probability:
        for index in range(len(child)):
            for i in range(len(child[index])):
                if len(np.unique(child[:, i, 1])) == 1:
                    new_chromosome = random.choice(chromosomes)
                    while not np.array_equal(new_chromosome, child[index, i]):
                        new_chromosome = random.choice(chromosomes)
                    child[index, i] = new_chromosome
    return child
```

### Selection Function
```python
def selection(index, scores):
    # Convert scores to non-negative values
    min_score = min(scores)
    non_negative_scores = scores - min_score + 1  # Add 1 to ensure non-negative scores
    distribution = non_negative_scores / non_negative_scores.sum()
    return choice(index, 2, p=distribution, replace=False)
```

### Crossover Function
```python
def crossover(parent1, parent2):
    if random.random() < 0.5:
        index = random.randint(1, len(parent1) - 2)
        child1 = np.concatenate((parent1[:index], parent2[index:]))
        child2 = np.concatenate((parent2[:index], parent1[index:]))
    else:
        index = random.randint(1, len(timeslots_theory) - 2)
        child1 = np.concatenate((parent1[:, :index], parent2[:, index:]), axis=1)
        child2 = np.concatenate((parent1[:, :index], parent2[:, index:]), axis=1)

    # Ensure new chromosomes are among the list of dictionaries)
    for i in range(len(child1)):
        for j in range(len(child1[i])):
            chromosome_index = np.where((chromosomes == child1[i, j]).all(axis=1))[0][0]
            if not np.array_equal(child1[i, j], chromosomes[chromosome_index]):
                child1[i, j] = random.choice(chromosomes)
            chromosome_index = np.where((chromosomes == child2[i, j]).all(axis=1))[0][0]
            if not np.array_equal(child2[i, j], chromosomes[chromosome_index]):
                child2[i, j] = random.choice(chromosomes)

    return [child1, child2]

```

### Generate Initial Population
```python
populationSize = min(len(rooms_theory), len(rooms_lab))
generations = 1000
mutationProb = 0.2

# Initialize empty timetables for each day of the week
timetables = {day: np.array([[[] for _ in range(len(timeslots_theory) + len(timeslots_lab))] for _ in range(populationSize)]) for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}

# Evolutionary loop for each day
for day in timetables.keys():
    # Separate populations for theory and lab
    theory_population = [np.array([[random.choice(theory_chromosomes) for _ in timeslots_theory] for _ in range(populationSize)]) for _ in range(populationSize)]
    lab_population = [np.array([[random.choice(lab_chromosomes) for _ in timeslots_lab] for _ in range(populationSize)]) for _ in range(populationSize)]
```


## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue to discuss improvements or bugs.
