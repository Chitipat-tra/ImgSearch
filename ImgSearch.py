import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

from google.cloud import discoveryengine_v1alpha
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct


# Constants for the API calls
PROJECT_ID = "721867696604"
DATA_STORE_ID = "villa-test-data-real_1702636375671"


@st.cache_resource
def get_client():
    return discoveryengine_v1alpha.SearchServiceClient()


def get_search_results(query, page_token=None):
    client = get_client()

    # Construct the request
    request = discoveryengine_v1alpha.SearchRequest(
        serving_config=f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_search:search",
        query=query,
        page_token=page_token,
        # filter=filter,
        query_expansion_spec=discoveryengine_v1alpha.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine_v1alpha.SearchRequest.QueryExpansionSpec.Condition.AUTO
        ),
        spell_correction_spec=discoveryengine_v1alpha.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine_v1alpha.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    # Use the search method from the client, passing the request object
    response = client.search(request=request)
    return response

@st.cache_data
def get_all_search_results(query):
    all_results = []

    # Call the search method to get the pager
    pager = get_search_results(query)
    all_results.extend(pager.results)
    while pager.next_page_token:
        pager = get_search_results(query, pager.next_page_token)
        all_results.extend(pager.results)
    return all_results

@st.cache_data
def generate_text_from_image(image_data):
    vertexai.init(project=PROJECT_ID, location="asia-southeast1")
    multimodal_model = GenerativeModel("gemini-pro-vision")

    response = multimodal_model.generate_content(
        [
            # "Analyze the image and identify the primary object relevant to supermarket items. Return a single, specific word that best describes this object for a search query. Format: '[single descriptive word]'.",
            # "Identify the object in the photo. Respond with only the essential descriptive nouns, including the type and brand if visible. Do not include any additional explanations. Format: '[Descriptive Noun with brand and type if visible]'",
            "Carefully examine the provided image to identify the primary object typically found in a supermarket. If the brand of the item is not clearly visible, leave the brand field empty. If the brand is visible, include it in your response. Follow these guidelines: If the brand is in Thai, provide both the brand name and the object exclusively in Thai language. If the brand is in English or no brand is visible, provide one word of the object in English. Ensure all responses are only in Thai or English language. Format your response as 'Brand: [brand name in the original language or empty if not visible], Object: [one word of the object in the same language as the brand or in English if the brand is not visible].' The result should be concise, accurate, and suitable for use in a search query.",
            Part.from_data(image_data, mime_type="image/jpeg"),
        ]
    )
    return response.text
 

def convert_struct_to_dict(struct_data):
    """Convert struct_data to a dictionary."""
    if struct_data:
        struct = Struct()
        struct.update(struct_data)
        return MessageToDict(struct)
    else:
        return {}

def extract_brand_and_object(full_description):
    # Split the description by commas
    parts = full_description.split(',')

    # Extract the brand and object
    brand_part = parts[0]  # Assumes the format is always "Brand: xxx"
    object_part = parts[1] if len(parts) > 1 else ''  # Assumes the format is "Object: xxx"

    # Clean up the strings and extract the necessary parts
    brand = brand_part.split('Brand: ')[1].strip() if 'Brand: ' in brand_part else 'Unknown'
    object_desc = object_part.split('Object: ')[1].strip() if 'Object: ' in object_part else 'Unknown'

    # Extract the last word of the object description
    last_word_of_object = object_desc.split()[-1] if object_desc else 'Unknown'
    # Return the brand and the last word of the object description
    return [brand, last_word_of_object]

@st.cache_data
def fetch_and_sort_results(brand, object_desc):
    all_results = []
    brand_results = []
    non_brand_results = []
    exact_match_results = []

    # Helper function for sorting
    def sort_key(item):
        return (
            item.document.struct_data.get("villa_category_l3_en", ""),
            item.document.struct_data.get("villa_category_l2_en", ""),
        )

    # Helper function to check brand match
    def is_brand_match(result, brand):
        if brand:
            fields_to_check = ["pr_name_en", "pr_name_th", "pr_brand_en", "pr_brand_th"]
            brand_lower = brand.lower()
            return any(brand_lower in result.document.struct_data.get(field, "").lower() for field in fields_to_check)
    
    # Helper function to check object exact match

    def is_object_exact_match(result, object_desc):
        object_lower = object_desc.lower()
    
        return object_lower == result.document.struct_data.get("content_en", "").lower() or \
               object_lower == result.document.struct_data.get("content_th", "").lower()
        
    # Perform object search and fetch all results
    all_results = get_all_search_results(object_desc)

    # Separate results based on brand match and object exact match
    for res in all_results:
        if is_brand_match(res, brand):
            brand_results.append(res)
        elif is_object_exact_match(res, object_desc):
            exact_match_results.append(res)
        else:
            non_brand_results.append(res)

    # Sort the results
    brand_results.sort(key=sort_key)
    exact_match_results.sort(key=sort_key)
    non_brand_results.sort(key=sort_key)
    # st.write(brand_results)
    # st.write(exact_match_results)
    # Concatenate sorted results: brand matches, exact object matches, then other non-brand matches
    combined_results = brand_results + exact_match_results + non_brand_results

    return combined_results


def prepare_dataframe(combined_results):
    df = pd.DataFrame([convert_struct_to_dict(item.document.struct_data) for item in combined_results])
    df.index = range(1, 1 + len(df))
    df.index.name = "Index"

    columns_order = [
                "content_en",
                "content_th",
                "villa_category_l3_en",
                "villa_category_l2_en",
            ] + [
                col
                for col in df.columns
                if col
                not in [
                    "content_en",
                    "content_th",
                    "villa_category_l3_en",
                    "villa_category_l2_en",
                ]
            ]
    df = df[columns_order]

    return df

def main():
    st.title("Image Search Application")
    query = ""
    uploaded_image = st.file_uploader(
        "Upload an image for search", type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        format = uploaded_image.type.split('/')[1].upper()
        if format not in ["JPEG", "PNG", "BMP"]:
            format = "JPEG"  # Fallback to JPEG if original format is not compatible
        with BytesIO() as img_byte_arr:
            image.save(img_byte_arr, format=format)
            image_data = img_byte_arr.getvalue()
        query = generate_text_from_image(image_data)
        st.write(f"Generated Query: {query}")

    if query:

        if "results_df" not in st.session_state or st.session_state.query != query:
            brand, object_desc = extract_brand_and_object(query)
            combined_results = fetch_and_sort_results(brand, object_desc)
            st.session_state.results_df = prepare_dataframe(combined_results)
            st.session_state.query = query
            st.session_state.page_number = 1

        # Pagination setup
        page_size = 10
        total_pages = len(st.session_state.results_df) // page_size + (len(st.session_state.results_df) % page_size > 0)
        offset = (st.session_state.page_number - 1) * page_size
        paginated_results = st.session_state.results_df.iloc[offset:offset + page_size]

        # Displaying results
        # Displaying results with inde
        st.table(paginated_results)  # Display DataFrame as a table without resetting the index
        st.write(
                f'Showing results {offset+1}-{min(offset+page_size, len(st.session_state.results_df))} out of {len(st.session_state.results_df)} for "{query}"'
                )
        # Navigation buttons below the table
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("First"):
                st.session_state.page_number = 1
        with col2:
            if st.button("Previous"):
                if st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
        with col3:
            if st.button("Next"):
                if st.session_state.page_number < total_pages:
                    st.session_state.page_number += 1
        with col4:
            if st.button("Last"):
                st.session_state.page_number = total_pages


if __name__ == "__main__":
    main()
