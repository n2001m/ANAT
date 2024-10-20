import cv2
import re
import numpy as np
import io
import langid
from google.cloud import vision
import json

# Load the Google vision model:
# Path to your JSON configuration file
json_path = ""

# Load configuration from JSON file
with open(json_path, 'r') as file:
    config = json.load(file)

# Initialize the Vision client
client = vision.ImageAnnotatorClient.from_service_account_json(json_path)


def preprocess_frame(frame, x1, y1, x2, y2):
    """
    Preprocess the given frame to crop the license plate, convert it to grayscale,
    and apply adaptive thresholding.

    Args:
        frame (numpy array): The input frame containing the car/license plate.
        x1, y1, x2, y2 (int): Bounding box coordinates of the license plate.

    Returns:
        license_plate_crop_thresh (numpy array): Preprocessed license plate image after adaptive thresholding.
    """
    # Crop the license plate from the frame using bounding box coordinates
    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

    # Convert the cropped license plate to grayscale
    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    license_plate_crop_thresh = cv2.adaptiveThreshold(
        license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )


# More loose assign_car:
def assign_car(license_plate, vehicle_track_ids, tolerance=0.1):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates with a margin of tolerance.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
        tolerance (float): Percentage of tolerance to make the bounding box check more lenient. Default is 10%.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    if len(license_plate) == 6:
        x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate
    elif len(license_plate) == 5:  # In case of missing class_id, default or skip
        x1_lp, y1_lp, x2_lp, y2_lp, score_lp = license_plate
        class_id_lp = None  # Handle the case of missing class_id

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        x1_v, y1_v, x2_v, y2_v, track_id, class_id_v = vehicle_track_ids[j]

        # Calculate the tolerance as a fraction of the vehicle's width and height
        width_v = x2_v - x1_v
        height_v = y2_v - y1_v
        tol_w = tolerance * width_v
        tol_h = tolerance * height_v

        # Check if the license plate is within the vehicle's bounding box with some tolerance
        if (x1_lp > (x1_v - tol_w) and y1_lp > (y1_v - tol_h) and
                x2_lp < (x2_v + tol_w) and y2_lp < (y2_v + tol_h)):
            car_indx = j
            foundIt = True  # If the car's license plate was found then break and assign it.
            break

    if foundIt:
        # Return the vehicle's bounding box and ID
        return vehicle_track_ids[car_indx][:5]
    # Return None if no matching vehicle is found
    return -1, -1, -1, -1, -1


# More strict implementation of assign_car:
# def assign_car(license_plate, vehicle_track_ids):
#     """
#     Retrieve the vehicle coordinates and ID based on the license plate coordinates.
#
#     Args:
#         license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
#         vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.
#
#     Returns:
#         tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
#     """
#     if len(license_plate) == 6:
#             x1_lp, y1_lp, x2_lp, y2_lp, score_lp, class_id_lp = license_plate
#     elif len(license_plate) == 5:  # In case of missing class_id, default or skip
#             x1_lp, y1_lp, x2_lp, y2_lp, score_lp = license_plate
#             class_id_lp = None  # Handle the case of missing class_id
#
#     foundIt = False
#     for j in range(len(vehicle_track_ids)):
#         xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j][:5]
#
#         if x1_lp > xcar1 and y1_lp > ycar1 and x2_lp < xcar2 and y2_lp < ycar2:
#             car_indx = j
#             foundIt = True
#             break
#
#     if foundIt:
#         return vehicle_track_ids[car_indx][:5]
#
#     return -1, -1, -1, -1, -1

def read_license_plate(cropped_plate):
    """
    Detect and format text from a cropped license plate using Google Vision API.

    Args:
        cropped_plate (Image): Cropped image containing the license plate.

    Returns:
        str: Formatted license plate text.
    """
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the plate.")
        return None, None

    # Extract all detected text
    all_text = texts[0].description if texts else ""

    # Filter for ASCII alphanumeric characters and spaces
    filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))

    # Split the text into parts
    parts = filtered_text.split()

    # Check if "KSA" is present
    if "KSA" in parts:
        # For plates with KSA, take the last part (should be the right side)
        plate_text = parts[-1]
    else:
        # For other plates, combine all parts, focusing on sequences of 3-4 characters
        plate_text = ' '.join(part for part in parts if len(part) >= 3 and len(part) <= 4)

    # Ensure we have both letters and numbers
    if not (any(c.isalpha() for c in plate_text) and any(c.isdigit() for c in plate_text)):
        # If not, try to extract letters and numbers separately
        letters = ''.join(c for c in filtered_text if c.isalpha())
        numbers = ''.join(c for c in filtered_text if c.isdigit())
        plate_text = f"{numbers} {letters}"  # Changed order here

    # Remove KSA if it's still present in the final plate_text
    plate_text = plate_text.replace("KSA", "").strip()

    # Split the plate text into numbers and letters
    numbers = ''.join(c for c in plate_text if c.isdigit())
    letters = ''.join(c for c in plate_text if c.isalpha())

    # If the number part has more than 4 digits, remove the last digit
    if len(numbers) > 4:
        numbers = numbers[:4]

    
    # Recombine numbers and letters
    plate_text = f"{numbers} {letters}"  # Changed order here

    print("Detected text in plate:", plate_text.strip())

    # Check if the filtered text complies with the Saudi license plate format
    if license_complies_format(plate_text):
        # Format the license plate according to the Saudi format (4 numbers-3 letters)
        formatted_plate = format_license(plate_text)
        confidence = texts[0].score if hasattr(texts[0], 'score') else 0.0
        return formatted_plate, confidence

    return None, None

def read_czech_license_plate(cropped_plate):
    """
    Detect and format text from a cropped license plate for the Czech Republic using Google Vision API.

    Args:
        cropped_plate (Image): Cropped image containing the license plate.

    Returns:
        tuple: (plate_text, confidence, plate_type)
    """
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    content = encoded_image.tobytes()

    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the plate.")
        return None, 0.0, None

    # Extract all detected text
    all_text = texts[0].description if texts else ""

    # Filter for ASCII alphanumeric characters, spaces, and hyphens
    filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace() or char == '-'))

    # Remove spaces and convert to uppercase
    plate_text = filtered_text.replace(" ", "").upper()

    # Define patterns for different types of Czech plates
    standard_pattern = re.compile(r'^([ABCEHIJKLMPSTUZ])(\d[A-Z]|[A-Z]\d|\d{2})\d{4}$')
    custom_pattern = re.compile(r'^[A-FHIJ-NP-TV-Z0-9]{3}-[A-FHIJ-NP-TV-Z0-9]{5}$')
    historic_pattern = re.compile(r'^[0-9]{2}V[0-9]{4}$')

    confidence = texts[0].confidence if hasattr(texts[0], 'confidence') else 0.0

    if standard_pattern.match(plate_text):
        return plate_text, confidence, "standard"
    elif custom_pattern.match(plate_text) and any(char.isdigit() for char in plate_text):
        return plate_text, confidence, "custom"
    elif historic_pattern.match(plate_text):
        return plate_text, confidence, "historic"
    else:
        print("Detected text does not match any known Czech license plate format.")
        return None, confidence, None

def global_read_license_plate(cropped_plate):
    """
    Detect and format text from a cropped license plate using Google Vision API.

    Args:
        cropped_plate (Image): Cropped image containing the license plate.

    Returns:
        str: Detected license plate text.
    """
    # Convert the cropped plate image to bytes
    success, encoded_image = cv2.imencode('.png', cropped_plate)
    if not success:
        print("Error encoding the image.")
        return None, None
    content = encoded_image.tobytes()

    # Create a Google Vision API image object
    image = vision.Image(content=content)

    # Perform text detection using the Vision API
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text detected in the plate.")
        return None, None

    # Extract the full text detected
    all_text = texts[0].description if texts else ""
    print(f"Detected raw text: {all_text}")

    # Filter for only alphanumeric characters and spaces (remove symbols, etc.)
    filtered_text = ''.join(char for char in all_text if char.isascii() and (char.isalnum() or char.isspace()))

    # Split the filtered text into parts (words)
    parts = filtered_text.split()

    # Combine parts that are 2-8 characters long, typically found on plates
    plate_candidates = [part for part in parts if 2 <= len(part) <= 8]

    # If no candidates are found, return None
    if not plate_candidates:
        print("No valid license plate candidates found.")
        return None, None

    # Join all valid parts into one string (final license plate text)
    plate_text = ' '.join(plate_candidates)

    # Ensure that the text contains both numbers and letters (common in license plates)
    if not (any(char.isalpha() for char in plate_text) and any(char.isdigit() for char in plate_text)):
        print("No valid alphanumeric combination detected in the plate.")
        return None, None

    # Optionally, compute confidence from the Vision API response (if available)
    confidence = texts[0].score if hasattr(texts[0], 'score') else 0.0

    print(f"Detected license plate: {plate_text.strip()}")
    
    # Return the detected plate text and the confidence score
    return plate_text.strip(), confidence

def write_csv(results, output_path):
    """
    Write the result to a CSV file.

    Args:
        results (dict): Dictionary containing the result.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_c', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


# Below is the code that would help the models predictions:

# Mapping dictionaries for character conversion
# Mapping characters that can be confused with numbers
dict_char_to_int = {
    'O': '0',  # O looks like 0
    'I': '1',  # I looks like 1
    'Z': '2',  # Z looks like 2
    'B': '8',  # B looks like 8
    'S': '5'  # S looks like 5
}

# Mapping numbers that can be confused with letters
dict_int_to_char = {
    '0': 'O',  # 0 can look like O
    '1': 'I',  # 1 can look like I
    '2': 'Z',  # 2 can look like Z
    '8': 'B',  # 8 can look like B
    '5': 'S'  # 5 can look like S
}


def license_complies_format(text):
    """
    Check if the license plate text complies with the Saudi license plate format (9999-XXX).

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Remove any potential dash in the license plate
    text = text.replace('-', '')

    # Check if the length is exactly 7 characters (4 digits + 3 letters)
    if len(text) != 7:
        return False

    # Check the first 4 characters are digits
    if not text[:4].isdigit():
        return False

    # Check the last 3 characters are uppercase letters
    if not text[4:].isalpha() or not text[4:].isupper():
        return False

    return True


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''

    # Mapping based on the position in the Saudi license plate format
    # Positions 0-3 are digits, positions 4-6 are letters
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char, 3: dict_int_to_char,
               4: dict_char_to_int, 5: dict_char_to_int, 6: dict_char_to_int}

    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    # Insert a dash between the numbers and letters (e.g., 1234-ABC)
    formatted_license_plate = license_plate_[:4] + '-' + license_plate_[4:]

    return formatted_license_plate


def filter_english_text(text):
    """
    Use langid to filter out non-English parts of the text.

    Args:
        text (str): The detected text from the license plate.

    Returns:
        str: Filtered English text.
    """
    # Split text into individual words or characters
    text_parts = text.split()

    # Filter for parts that are detected as English by langid
    english_text = ' '.join(part for part in text_parts if langid.classify(part)[0] == 'en')

    return english_text


# New format_license_re function using regex
def format_license_re(license_text):
    # Regex pattern to match a standard Saudi license plate pattern (modify as needed)
    # Assuming Saudi plates follow the format: 3 digits + 3 letters (Example: 123ABC)
    pattern = r"(\d{1,3})([A-Za-z]{1,3})"

    match = re.match(pattern, license_text)
    if match:
        digits, letters = match.groups()
        return f"{digits}-{letters}".upper()
    else:
        return license_text.upper()


# Check and replace unaccepted characters for Saudi license plates
def validate_license_plate(license_text):
    # Saudi plates generally allow A-Z (some limited set of letters, adjust accordingly)
    allowed_letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'  # Exclude I, O, Q for example
    allowed_digits = '0123456789'

    valid_license = []

    for char in license_text:
        if char in allowed_digits:
            valid_license.append(char)
        elif char in allowed_letters:
            valid_license.append(char)
        else:
            # Replace with a placeholder or closest valid character
            valid_license.append('_')  # Placeholder or handle replacement logic here

    return ''.join(valid_license)
