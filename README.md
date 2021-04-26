# Collision-Classification-and-Matching
- This project was the final project for course `Visual and Acoustic Information System`, aka `视听导`, 2020 fall.
- This project was led by `Anbai Jiang`, contributed by `A L` and `Xt X`.

## Tasks

- The whole project can be divided into 3 parts. For detailed information, pleas turn to  `大作业要求.pdf`
- **Dataset**
  - This dataset was an open-source project from `CMU`.
  - Objects are placed on an unstable platform that can create random vibrations, rendering the object to hit the bound.
  - A camera captures the collision video. Four microphones captures the collision audio.
- **Task 1**: Object classification based on audio.
- **Task 2**: Full match between 50 video segments and 50 audio segments.
  - Output a mapping between which video and which audio.
- **Task 3**: Best match between 50 videos segments and 50 audio segments.
  - There are distracting videos and audios that won't match.

## Implementations

- **Task 1**

  - `MFCC` for audio feature extraction.
  - `CNN` for audio feature classification
  - Average accuracy on train and validation: 93%

- **Task 2**

  - `CNN` for video classification, no need for extra feature extraction.

    - Average accuracy on train and validation: 100%

  - Do classification on both videos and audios first, group those belonging to the same category. 

  - Then do matching within each group.

    - 2 features for matching 

      - |            |              feature 1              |                feature 2                |
        | :--------: | :---------------------------------: | :-------------------------------------: |
        |   video    |       maximum speed of object       |     collision edge based on vision      |
        |   audio    |   maximum amplitude in all 4 mics   | collision edge based on audio amplitude |
        | philosophy | Faster speed leads to louder noise. | Detect which edged the object collides. |

    - Define match loss based on these 2 features, polynomial, reinforced with a confidence score.

    - Full match. Output the mapping which has the lowest match loss.

  - Unmatched source from all groups do a match again after the main match within each group.

- **Task 3**

  - Most are similar to task 2, except for match strategy: greedy.

