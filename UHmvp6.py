import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Ultrahuman Metabolic Score",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .score-display {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .green-score {
        background-color: #4CAF50;
        color: white;
    }
    .light-green-score {
        background-color: #8BC34A;
        color: white;
    }
    .yellow-score {
        background-color: #FFC107;
        color: #333;
    }
    .red-score {
        background-color: #F44336;
        color: white;
    }
    .recommendation-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .food-card {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to safely parse glucose values
def parse_glucose_values(val_str):
    if isinstance(val_str, str):
        try:
            # Remove brackets and split by comma
            return ast.literal_eval(val_str)
        except:
            # If fails, try a simpler approach
            values = val_str.strip('[]').split(',')
            return [float(v.strip()) for v in values]
    elif isinstance(val_str, list):
        return val_str
    return []

# Function to generate glucose response curve
def generate_glucose_curve(baseline, glucose_peak, recovery_time, minutes=180):
    """Generate a realistic glucose response curve"""
    time_points = np.arange(0, minutes, 5)  # 5-min intervals
    glucose_values = []
    
    for t in time_points:
        if t < 10:  # Initial lag
            value = baseline
        elif t < 35:  # Rise phase - steeper for higher peaks
            rise_factor = min(1.0, (t-10)/25)
            # Smooth S-curve shape for more realistic rise
            rise_shape = 3 * rise_factor**2 - 2 * rise_factor**3
            value = baseline + (glucose_peak * rise_shape)
        else:  # Decay phase - slower for higher peaks
            time_since_peak = t - 35
            decay_factor = np.exp(-time_since_peak / (recovery_time * (1 + glucose_peak/100)))
            value = baseline + glucose_peak * decay_factor
            
        # Add small random noise
        value += np.random.normal(0, 0.8)
        glucose_values.append(round(value, 1))
        
    return time_points, glucose_values

# Function to calculate metabolic score
def calculate_metabolic_score(first_food_type, total_carbs, total_protein, total_fat, total_fiber, max_gi, glucose_peak):
    """Calculate metabolic score based on food composition and sequence"""
    base_score = 75  # Start with a moderate score
    
    # Sequence factor - major impact on score
    if first_food_type in ['vegetable', 'protein']:
        sequence_bonus = 15
    elif first_food_type in ['carb', 'sweet']:
        sequence_bonus = -15
    elif first_food_type == 'fruit':
        sequence_bonus = -8
    elif first_food_type in ['fat', 'dairy']:
        sequence_bonus = 5
    else:
        sequence_bonus = 0
        
    # Composition adjustments
    if total_fiber > 5:
        base_score += 5
    if total_protein > 20:
        base_score += 5
    if total_carbs > 50:
        base_score -= 5
    if max_gi > 70:
        base_score -= 3
        
    # Glucose impact
    if glucose_peak < 20:
        base_score += 5
    elif glucose_peak > 40:
        base_score -= 5
        
    # Apply sequence bonus
    final_score = base_score + sequence_bonus
    
    # Ensure score is within 1-100 range
    final_score = max(1, min(100, final_score))
    
    return int(final_score)

# Function to get score color
def get_score_color(score):
    if score >= 85:
        return "green-score"
    elif score >= 70:
        return "light-green-score"
    elif score >= 50:
        return "yellow-score"
    else:
        return "red-score"

# Function to load data
@st.cache_data
def load_data():
    try:
        # Define the base data directory
        base_dir = "data/"
        
        # Load food database
        food_db = pd.read_csv(f'{base_dir}extended_food_database.csv')
        
        # Load glucose responses
        glucose_responses = pd.read_csv(f'{base_dir}extended_glucose_responses.csv')
        
        # Load meal combinations
        meal_combinations = pd.read_csv(f'{base_dir}extended_meal_combinations.csv')
        
        # Load meal foods
        meal_foods = pd.read_csv(f'{base_dir}extended_meal_foods.csv')
        
        # Generate food type statistics from glucose responses if needed
        # If we can extract first_food_type from meal combinations, we'll use that
        if 'food_sequence' in meal_combinations.columns:
            # Extract first food type from sequences
            meal_combinations['first_food_type'] = meal_combinations['food_sequence'].apply(
                lambda x: x.split(' → ')[0] if isinstance(x, str) and ' → ' in x else x
            )
            
            # Join with glucose responses
            if 'meal_id' in glucose_responses.columns and 'meal_id' in meal_combinations.columns:
                merged_data = pd.merge(
                    glucose_responses,
                    meal_combinations[['meal_id', 'first_food_type']],
                    on='meal_id',
                    how='inner'
                )
                
                # Calculate statistics by food type
                if 'metabolic_score' in merged_data.columns and 'glucose_elevation' in merged_data.columns:
                    food_type_stats = merged_data.groupby('first_food_type').agg({
                        'metabolic_score': ['mean', 'min', 'max', 'count'],
                        'glucose_elevation': ['mean', 'min', 'max']
                    }).reset_index()
                    
                    # Flatten multi-level columns
                    food_type_stats.columns = ['_'.join(col).strip('_') for col in food_type_stats.columns.values]
                    
                    # Round numeric values
                    for col in food_type_stats.columns:
                        if col != 'first_food_type' and 'count' not in col:
                            food_type_stats[col] = food_type_stats[col].round(1)
                else:
                    # Fallback if columns don't exist
                    food_type_stats = pd.DataFrame({
                        'first_food_type': ['vegetable', 'protein', 'fat', 'dairy', 'fruit', 'carb', 'sweet'],
                        'metabolic_score_mean': [95, 94, 90, 89, 72, 64, 62],
                        'glucose_elevation_mean': [12.9, 15.4, 10.6, 8.2, 29.8, 38.5, 43.0]
                    })
            else:
                # Fallback if merge isn't possible
                food_type_stats = pd.DataFrame({
                    'first_food_type': ['vegetable', 'protein', 'fat', 'dairy', 'fruit', 'carb', 'sweet'],
                    'metabolic_score_mean': [95, 94, 90, 89, 72, 64, 62],
                    'glucose_elevation_mean': [12.9, 15.4, 10.6, 8.2, 29.8, 38.5, 43.0]
                })
        else:
            # Fallback if food_sequence column doesn't exist
            food_type_stats = pd.DataFrame({
                'first_food_type': ['vegetable', 'protein', 'fat', 'dairy', 'fruit', 'carb', 'sweet'],
                'metabolic_score_mean': [95, 94, 90, 89, 72, 64, 62],
                'glucose_elevation_mean': [12.9, 15.4, 10.6, 8.2, 29.8, 38.5, 43.0]
            })
        
        # Sort by metabolic score
        food_type_stats = food_type_stats.sort_values('metabolic_score_mean', ascending=False)
        
        # Create recommended sequence
        recommended_sequence = food_type_stats[['first_food_type']].copy()
        if 'metabolic_score_mean' in food_type_stats.columns:
            recommended_sequence['metabolic_score_mean'] = food_type_stats['metabolic_score_mean']
        else:
            recommended_sequence['metabolic_score_mean'] = [95, 94, 90, 89, 72, 64, 62][:len(recommended_sequence)]
            
        recommended_sequence['recommended_order'] = range(1, len(recommended_sequence) + 1)
        
        # Create basic food recommendations
        food_recommendations = pd.DataFrame()
        
        return food_db, food_type_stats, recommended_sequence, food_recommendations, glucose_responses, meal_combinations
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide simple fallback data if files don't exist
        return None, None, None, None, None, None

# Load data
food_db, food_type_stats, recommended_sequence, food_recommendations, glucose_responses, meal_combinations = load_data()

# Check if data is loaded successfully
if food_db is None:
    st.error("Failed to load food database. Please check that the exported files are in the data/ directory.")
    
    # Show detailed debug information
    st.write("Looking for files in: data/")
    st.write("Expected filenames:")
    st.write("- extended_food_database.csv")
    st.write("- extended_glucose_responses.csv")
    st.write("- extended_meal_combinations.csv")
    st.write("- extended_meal_foods.csv")
    
    # Try to list files in the current directory
    try:
        import os
        files_in_current_dir = os.listdir('.')
        st.write("Files in current directory:", files_in_current_dir)
        
        if 'data' in files_in_current_dir:
            files_in_data_dir = os.listdir('data')
            st.write("Files in data/ directory:", files_in_data_dir)
    except Exception as e:
        st.write(f"Error listing directory contents: {e}")
    
    st.stop()

# App title and introduction
st.title("Ultrahuman Metabolic Score")
st.subheader("Enhance your metabolic health through optimized food sequencing")

st.markdown("""
This tool demonstrates how the order in which you eat foods affects your glucose response and metabolic health.
By optimizing your food sequence, you can significantly reduce glucose spikes and improve your overall metabolic score.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Food Sequence Impact", "Meal Builder", "About"])

# Tab 1: Food Sequence Impact
with tab1:
    st.header("The Impact of Food Sequence")
    
    st.markdown("""
    Research shows that the order in which you consume foods can significantly impact your glucose response.
    Starting your meal with proteins or vegetables before carbohydrates can reduce glucose spikes by up to 40%.
    """)
    
    # Sort food types by metabolic score
    sorted_types = food_type_stats.sort_values('metabolic_score_mean', ascending=False)
    
    # Display visualizations side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metabolic Score by First Food Type")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(sorted_types['first_food_type'], sorted_types['metabolic_score_mean'])
        
        # Color bars based on score
        for i, bar in enumerate(bars):
            score = sorted_types.iloc[i]['metabolic_score_mean']
            if score >= 85:
                bar.set_color('#4CAF50')  # green
            elif score >= 70:
                bar.set_color('#8BC34A')  # light green
            elif score >= 50:
                bar.set_color('#FFC107')  # yellow
            else:
                bar.set_color('#F44336')  # red
                
        ax.set_xlabel('First Food Type in Sequence')
        ax.set_ylabel('Average Metabolic Score')
        ax.set_title('Impact of First Food Type on Metabolic Score')
        ax.set_ylim(0, 100)
        
        # Add score labels
        for i, v in enumerate(sorted_types['metabolic_score_mean']):
            ax.text(i, v + 1, f"{v:.1f}", ha='center')
            
        st.pyplot(fig)
    
    with col2:
        st.subheader("Glucose Response by First Food Type")
        
        # Sort by glucose elevation
        glucose_sorted = food_type_stats.sort_values('glucose_elevation_mean')
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(glucose_sorted['first_food_type'], glucose_sorted['glucose_elevation_mean'])
        
        # Color bars based on elevation
        for i, bar in enumerate(bars):
            elevation = glucose_sorted.iloc[i]['glucose_elevation_mean']
            if elevation < 15:
                bar.set_color('#4CAF50')  # green
            elif elevation < 25:
                bar.set_color('#8BC34A')  # light green
            elif elevation < 35:
                bar.set_color('#FFC107')  # yellow
            else:
                bar.set_color('#F44336')  # red
                
        ax.set_xlabel('First Food Type in Sequence')
        ax.set_ylabel('Average Glucose Elevation (mg/dL)')
        ax.set_title('Glucose Response by First Food Type')
        
        # Add elevation labels
        for i, v in enumerate(glucose_sorted['glucose_elevation_mean']):
            ax.text(i, v + 1, f"{v:.1f}", ha='center')
            
        st.pyplot(fig)
    
    # Display glucose curves
    st.subheader("Glucose Response Curves")
    
    # Generate example glucose curves for different first food types
    baseline = 85
    time_points = np.arange(0, 180, 5)
    
    # Define parameters for different food types
    curve_params = {
        'vegetable': {'peak': 12, 'recovery': 30, 'color': '#4CAF50', 'score': 95},
        'protein': {'peak': 15, 'recovery': 35, 'color': '#8BC34A', 'score': 94},
        'fruit': {'peak': 30, 'recovery': 45, 'color': '#FFC107', 'score': 72},
        'carb': {'peak': 38, 'recovery': 60, 'color': '#F44336', 'score': 64}
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for food_type, params in curve_params.items():
        time_points, glucose_values = generate_glucose_curve(baseline, params['peak'], params['recovery'])
        ax.plot(time_points[:len(glucose_values)], glucose_values, 
                label=f"{food_type} first (Score: {params['score']})",
                color=params['color'],
                linewidth=3, alpha=0.8)
    
    # Add a horizontal line for normal glucose
    ax.axhline(y=85, color='gray', linestyle='--', alpha=0.7, label='Baseline glucose')
    
    # Add shaded regions for glucose ranges
    ax.axhspan(70, 85, alpha=0.1, color='green', label='Ideal range')
    ax.axhspan(85, 140, alpha=0.1, color='yellow')
    ax.axhspan(140, 200, alpha=0.1, color='red', label='Elevated')
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
    ax.set_title('Effect of First Food Type on Glucose Response Curve', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    
    # Annotations to explain the impact
    ax.annotate('Lower peaks = better metabolic health',
                xy=(35, 120), xytext=(60, 150),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12)
                
    st.pyplot(fig)
    
    # Display food sequence guidance
    st.subheader("Recommended Food Sequence")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Optimal Sequence:
        1. **Vegetables** (Non-starchy)
        2. **Proteins** (Meat, fish, tofu, etc.)
        3. **Fats** (Nuts, seeds, oils)
        4. **Dairy** (Yogurt, cheese)
        5. **Fruits** (Whole, not juiced)
        6. **Carbohydrates** (Grains, starches)
        7. **Sweets** (If consumed)
        """)
    
    with col2:
        st.markdown("""
        ### Key Benefits:
        - **Lower glucose spikes** by 30-50%
        - **Improved insulin sensitivity**
        - **Better satiety** and reduced cravings
        - **Slower gastric emptying**
        - **Reduced fat storage**
        - **Improved energy levels**
        """)
    
    # Display data table
    st.subheader("Food Sequence Impact Data")
    
    # Create a more informative table
    impact_data = pd.DataFrame({
        'First Food Type': sorted_types['first_food_type'].str.capitalize(),
        'Average Metabolic Score': sorted_types['metabolic_score_mean'],
        'Average Glucose Elevation (mg/dL)': sorted_types['glucose_elevation_mean'],
        'Recommended Sequence Position': range(1, len(sorted_types) + 1)
    })
    
    st.dataframe(impact_data, use_container_width=True)
    
    st.markdown("""
    **Key Insight:** Starting your meal with vegetables or protein results in a significantly higher metabolic score and
    lower glucose elevation compared to starting with carbohydrates or sweets.
    """)
    
    # Try to display actual glucose curves from the data
    actual_curves_shown = False
    
    try:
        # Use the correct function name that's defined earlier
        if glucose_responses is not None and 'glucose_values' in glucose_responses.columns:
            st.subheader("Actual Glucose Responses from Your Data")
            
            # Get some sample responses
            # Try to get different first food types
            sample_responses = []
            food_types = ['vegetable', 'protein', 'carb', 'fruit']
            
            # Find examples of each food type if available
            for food_type in food_types:
                # Find meal IDs for this food type
                if meal_combinations is not None and 'first_food_type' in meal_combinations.columns:
                    matching_meals = meal_combinations[meal_combinations['first_food_type'] == food_type]
                    if len(matching_meals) > 0:
                        meal_id = matching_meals.iloc[0]['meal_id']
                        # Find this meal's response
                        matching_response = glucose_responses[glucose_responses['meal_id'] == meal_id]
                        if len(matching_response) > 0:
                            sample_responses.append(matching_response.iloc[0])
            
            # If we found samples, create the visualization
            if len(sample_responses) > 0:
                # Create the plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Color map for different food types
                colors = {
                    'vegetable': '#4CAF50',
                    'protein': '#8BC34A',
                    'fruit': '#FFC107',
                    'carb': '#F44336',
                    'sweet': '#9C27B0',
                    'fat': '#2196F3',
                    'dairy': '#00BCD4'
                }
                
                # Plot each sample
                for i, response in enumerate(sample_responses):
                    # Get meal ID and find first food type
                    meal_id = response['meal_id']
                    
                    # Find first food type from meal combinations
                    first_food_type = "unknown"
                    if meal_combinations is not None:
                        matching_meal = meal_combinations[meal_combinations['meal_id'] == meal_id]
                        if len(matching_meal) > 0 and 'first_food_type' in matching_meal.columns:
                            first_food_type = matching_meal.iloc[0]['first_food_type']
                    
                    # Parse glucose values
                    if 'glucose_values' in response:
                        glucose_values = parse_glucose_values(response['glucose_values'])
                        
                        # Get score if available
                        score = response.get('metabolic_score', 0)
                        
                        # Generate time points
                        time_points = np.arange(0, len(glucose_values) * 5, 5)[:len(glucose_values)]
                        
                        # Plot the curve
                        color = colors.get(first_food_type, 'gray')
                        ax.plot(time_points, glucose_values, 
                                label=f"{first_food_type} first (Score: {score})",
                                color=color,
                                linewidth=3, alpha=0.8)
                
                # Add baseline, labels, etc.
                baseline = 85  # Approximate baseline
                ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label='Baseline glucose')
                ax.set_xlabel('Time (minutes)', fontsize=12)
                ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
                ax.set_title('Actual Glucose Responses by First Food Type', fontsize=16)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(fontsize=10)
                
                st.pyplot(fig)
                actual_curves_shown = True
    except Exception as e:
        st.warning(f"Could not display actual glucose curves: {e}")
    
    if not actual_curves_shown:
        st.info("Note: No actual glucose curves were found in your data. The graphs above are based on simulated responses derived from your data analysis.")

# Tab 2: Meal Builder
with tab2:
    st.header("Build Your Optimal Meal Sequence")
    
    st.markdown("""
    Select foods to build your meal, and see how different sequences affect your glucose response
    and metabolic score. The tool will suggest the optimal sequence for the foods you select.
    """)
    
    # Get available categories first
    available_categories = sorted(food_db['food_category'].unique())
    
    # Initialize session state for selected foods and filters
    if 'selected_food_ids' not in st.session_state:
        st.session_state.selected_food_ids = []
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = available_categories[:3]
    
    # Initialize session state for filters
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = available_categories[:3]
    
    # Create a row for the Reset buttons
    reset_col1, reset_col2 = st.columns([1, 1])
    
    with reset_col1:
        # Button to reset all selections
        if st.button("Reset All Selections", key="reset_button"):
            st.session_state.selected_food_ids = []
            st.experimental_rerun()
    
    with reset_col2:
        # Button to clear filters
        if st.button("Clear All Filters", key="clear_filters"):
            st.session_state.search_query = ""
            st.session_state.selected_categories = available_categories
            st.experimental_rerun()
    
    # Filter explanation
    st.info("Use filters to find foods to add to your meal. Your selected foods will be shown below regardless of current filter settings.")
    
    # Create a multi-select for food categories with session state
    selected_categories = st.multiselect(
        "Filter by food category",
        available_categories,
        default=st.session_state.selected_categories
    )
    
    # Update session state
    st.session_state.selected_categories = selected_categories
    
    # Create search box with session state
    search_query = st.text_input("Search for a specific food", value=st.session_state.search_query)
    
    # Update session state
    st.session_state.search_query = search_query
    
    # Apply filters
    filtered_foods = food_db
    
    # Apply category filter if selections were made
    if selected_categories:
        filtered_foods = filtered_foods[filtered_foods['food_category'].isin(selected_categories)]
    
    # Apply search filter if a query was entered
    if search_query:
        filtered_foods = filtered_foods[filtered_foods['food_name'].str.contains(search_query, case=False)]
        
    # If filters result in no foods, show a helpful message
    if len(filtered_foods) == 0:
        st.warning(f"No foods match your current filters. Try adjusting your search criteria or category selection.")
        
        # Show what's been selected but filtered out
        if st.session_state.selected_food_ids:
            st.subheader("Your currently selected foods")
            selected_food_names = []
            
            for food_id in st.session_state.selected_food_ids:
                matching_foods = food_db[food_db['food_id'] == food_id]
                if not matching_foods.empty:
                    food = matching_foods.iloc[0]
                    selected_food_names.append(f"{food['food_name']} ({food['food_category']})")
            
            if selected_food_names:
                for name in selected_food_names:
                    st.write(f"• {name}")
                st.info("Clear filters to see all your selections.")
    
    st.subheader("Step 1: Select foods for your meal")
    
    # Group foods by category
    categories_to_display = filtered_foods['food_category'].unique()
    
    selected_foods = []
    
    # Handle empty results gracefully
    if len(categories_to_display) == 0:
        st.warning("No foods match your current filters. Try adjusting your search criteria or category selection.")
    else:
        # Create columns based on number of categories (safely)
        num_cols = min(len(categories_to_display), 3)
        if num_cols > 0:  # Make sure we have at least one column
            cols = st.columns(num_cols)
            
            # Display foods by category
            for i, category in enumerate(sorted(categories_to_display)):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    st.markdown(f"**{category}**")
                    
                    category_foods = filtered_foods[filtered_foods['food_category'] == category]
                    
                    for _, food in category_foods.iterrows():
                        food_id = food['food_id']
                        food_info = f"{food['food_name']} ({food['carbs']}g carbs, {food['protein']}g protein)"
                        
                        # Use the session state to maintain checkbox state
                        checkbox_key = f"food_{food_id}"
                        is_checked = checkbox_key in st.session_state and st.session_state[checkbox_key]
                        
                        # Check if this food_id is in our selected_food_ids list
                        is_selected = food_id in st.session_state.selected_food_ids
                        
                        # Create checkbox with the current state
                        if st.checkbox(food_info, value=is_selected, key=checkbox_key):
                            # If checked and not already in the list, add it
                            if food_id not in st.session_state.selected_food_ids:
                                st.session_state.selected_food_ids.append(food_id)
                            selected_foods.append(food.to_dict())
                        else:
                            # If unchecked and in the list, remove it
                            if food_id in st.session_state.selected_food_ids:
                                st.session_state.selected_food_ids.remove(food_id)
    
    # If no foods are manually selected but we have foods in session state,
    # load those foods from the database (even when filtered)
    if len(st.session_state.selected_food_ids) > 0:
        # Reset selected_foods to ensure we get all selected foods
        selected_foods = []
        for food_id in st.session_state.selected_food_ids:
            matching_foods = food_db[food_db['food_id'] == food_id]
            if not matching_foods.empty:
                selected_foods.append(matching_foods.iloc[0].to_dict())
    
    # Display a summary of all selected foods regardless of current filter
    if selected_foods:
        st.subheader("Your selected foods")
        
        # Create columns for the food list
        food_cols = st.columns(3)
        
        # Display selected foods with their categories
        for i, food in enumerate(selected_foods):
            col_idx = i % 3
            with food_cols[col_idx]:
                st.markdown(f"**{food['food_name']}** ({food['food_category']})")
        
        st.write("")  # Add some spacing
    
    if selected_foods:
        st.subheader("Step 2: Arrange your meal sequence")
        
        # Display selected foods with drag handles
        st.markdown("Drag foods to arrange them in the order you'll eat them:")
        
        # Create a mapping of positions for each food
        positions = {}
        available_positions = list(range(1, len(selected_foods) + 1))
        
        cols = st.columns(len(selected_foods))
        
        for i, food in enumerate(selected_foods):
            with cols[i]:
                st.markdown(f"**{food['food_name']}**")
                position = st.selectbox("Position", available_positions, key=f"pos_{food['food_id']}")
                positions[food['food_id']] = position
        
        # Sort foods by position
        sequence_foods = sorted(selected_foods, key=lambda x: positions[x['food_id']])
        
        # Calculate meal metrics
        total_carbs = sum(food['carbs'] for food in sequence_foods)
        total_protein = sum(food['protein'] for food in sequence_foods)
        total_fat = sum(food['fat'] for food in sequence_foods)
        total_fiber = sum(food['fiber'] for food in sequence_foods)
        max_gi = max(food['gi_value'] for food in sequence_foods) if sequence_foods else 0
        
        # Extract food types in sequence
        food_types = [food['food_type'] for food in sequence_foods]
        
        # Display the sequence
        st.subheader("Your meal sequence:")
        sequence_str = " → ".join([food['food_name'] for food in sequence_foods])
        st.markdown(f"**{sequence_str}**")
        
        # Calculate current sequence glucose impact
        first_food_type = food_types[0] if food_types else ''
        
        # Calculate glucose response parameters
        if first_food_type in ['carb', 'sweet']:
            base_factor = 1.5
        elif first_food_type == 'fruit':
            base_factor = 1.3
        elif first_food_type in ['protein', 'vegetable']:
            base_factor = 0.6
        else:
            base_factor = 1.0
        
        # Simple formula for glucose peak
        glucose_peak = (total_carbs * max_gi/100 * 0.4) * base_factor
        glucose_peak = max(glucose_peak, 15)  # Ensure minimum peak
        
        # Recovery time based on carbs and sequence
        recovery_time = (glucose_peak * 0.5) + 30
        
        # Calculate metabolic score
        current_score = calculate_metabolic_score(
            first_food_type, total_carbs, total_protein,
            total_fat, total_fiber, max_gi, glucose_peak
        )
        
        # Create optimal sequence
        food_type_order = ['vegetable', 'protein', 'fat', 'dairy', 'fruit', 'carb', 'sweet']
        
        # Group foods by type
        foods_by_type = {}
        for food in selected_foods:
            food_type = food['food_type']
            if food_type not in foods_by_type:
                foods_by_type[food_type] = []
            foods_by_type[food_type].append(food)
        
        # Create the optimal sequence
        optimal_sequence = []
        for food_type in food_type_order:
            if food_type in foods_by_type:
                optimal_sequence.extend(foods_by_type[food_type])
        
        # Display optimal sequence
        optimal_sequence_str = " → ".join([food['food_name'] for food in optimal_sequence])
        st.markdown(f"**Recommended optimal sequence:** {optimal_sequence_str}")
        
        # Calculate optimal sequence glucose impact
        optimal_first_type = optimal_sequence[0]['food_type'] if optimal_sequence else ''
        
        # Calculate optimal glucose response
        if optimal_first_type in ['protein', 'vegetable']:
            optimal_factor = 0.6
        elif optimal_first_type == 'fruit':
            optimal_factor = 1.3
        elif optimal_first_type in ['carb', 'sweet']:
            optimal_factor = 1.5
        else:
            optimal_factor = 1.0
        
        # Calculate optimal glucose peak
        optimal_glucose_peak = (total_carbs * max_gi/100 * 0.4) * optimal_factor
        optimal_glucose_peak = max(optimal_glucose_peak, 12)
        
        # Calculate optimal recovery time
        optimal_recovery_time = (optimal_glucose_peak * 0.5) + 30
        
        # Calculate optimal score
        optimal_score = calculate_metabolic_score(
            optimal_first_type, total_carbs, total_protein,
            total_fat, total_fiber, max_gi, optimal_glucose_peak
        )
        
        # Display results
        st.subheader("Impact on Metabolic Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Display current score with color
            current_color = get_score_color(current_score)
            
            st.markdown(
                f"""
                <div class="score-display {current_color}">
                {current_score}
                </div>
                <p style="text-align:center">Your Metabolic Score</p>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            # Display optimal score with color
            optimal_color = get_score_color(optimal_score)
            
            st.markdown(
                f"""
                <div class="score-display {optimal_color}">
                {optimal_score}
                </div>
                <p style="text-align:center">Optimal Metabolic Score</p>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            # Show improvement metrics
            improvement = optimal_score - current_score
            glucose_reduction = glucose_peak - optimal_glucose_peak
            
            st.metric("Score Improvement", f"+{improvement}" if improvement > 0 else f"{improvement}")
            st.metric("Glucose Peak Reduction", f"-{glucose_reduction:.1f} mg/dL" if glucose_reduction > 0 else "0 mg/dL")
            
            if optimal_first_type != first_food_type:
                st.markdown(f"**Key Change:** Start with **{optimal_first_type}** instead of **{first_food_type}**")
        
        # Generate glucose curves for comparison
        st.subheader("Glucose Response Comparison")
        
        # Generate time points
        time_points = np.arange(0, 180, 5)
        
        # Generate glucose curves
        _, current_curve = generate_glucose_curve(85, glucose_peak, recovery_time)
        _, optimal_curve = generate_glucose_curve(85, optimal_glucose_peak, optimal_recovery_time)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot current sequence
        ax.plot(time_points[:len(current_curve)], current_curve, 
                label=f"Current sequence (Score: {current_score})",
                alpha=0.7)
        
        # Plot optimal sequence
        ax.plot(time_points[:len(optimal_curve)], optimal_curve, 
                label=f"Optimal sequence (Score: {optimal_score})",
                color="#4CAF50" if optimal_score >= 85 else "#FFC107")
        
        # Add baseline
        ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Baseline glucose')
        
        # Add labels and legend
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title('Comparison of Glucose Responses')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Display the plot
        st.pyplot(fig)
        
        # Nutritional breakdown
        st.subheader("Meal Nutrition Breakdown")
        
        # Create columns for metrics
        metric_cols = st.columns(5)
        
        with metric_cols[0]:
            st.metric("Total Carbs", f"{total_carbs:.1f}g")
        
        with metric_cols[1]:
            st.metric("Total Protein", f"{total_protein:.1f}g")
        
        with metric_cols[2]:
            st.metric("Total Fat", f"{total_fat:.1f}g")
        
        with metric_cols[3]:
            st.metric("Total Fiber", f"{total_fiber:.1f}g")
        
        with metric_cols[4]:
            st.metric("Max GI Value", f"{max_gi}")
        
        # Recommendations
        st.subheader("Personalized Recommendations")
        
        recommendations = []
        
        # Sequence recommendations
        if first_food_type in ['carb', 'sweet', 'fruit']:
            recommendations.append("**Sequence Optimization:** Start your meal with vegetables or protein before consuming carbs or sweets")
        
        # Fiber recommendations
        if total_fiber < 5:
            recommendations.append("**Increase Fiber:** Consider adding more high-fiber foods to your meal")
        
        # Protein recommendations
        if total_protein < 15:
            recommendations.append("**Increase Protein:** Adding more protein can help moderate glucose response")
        
        # High GI recommendations
        if max_gi > 70:
            recommendations.append("**High GI Foods:** Consider pairing high-glycemic foods with protein, fat, or fiber")
        
        # Display recommendations
        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"""
                <div class="recommendation-box">
                {recommendation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Your meal is well-optimized for metabolic health! Keep up the good work.")
    
    else:
        st.info("Please select at least one food to build your meal sequence")

# Tab 3: About
with tab3:
    st.header("About This Tool")
    
    st.markdown("""
    ## What is Metabolic Health?
    
    Metabolic health is a state where your body efficiently processes food into energy.
    It's influenced by factors like glucose regulation, insulin sensitivity, and inflammation.
    Poor metabolic health can lead to conditions like diabetes, obesity, and heart disease.
    
    ## The Science Behind Food Sequencing
    
    Multiple clinical studies have shown that the order in which you consume different food types
    significantly impacts glucose response:
    
    - A 2015 study in Diabetes Care found that consuming protein and vegetables before carbohydrates
    reduced glucose peaks by 29-37% compared to eating the same foods in reverse order
    
    - Research shows consuming fiber and protein before carbohydrates slows digestion and reduces glucose spikes
    
    - The "vegetables first" approach has been shown to improve satiety and reduce overall calorie intake
    
    ## How This Tool Works
    
    The Ultrahuman Metabolic Score tool uses a sophisticated algorithm that:
    
    1. Analyzes the composition of meals (carbs, protein, fat, fiber, glycemic index)
    
    2. Evaluates the sequence in which foods are consumed
    
    3. Predicts glucose response patterns based on food order and composition
    
    4. Calculates a metabolic score that reflects the healthiness of the glucose response
    
    5. Provides personalized recommendations to optimize meal sequences
    
    ## Building for Ultrahuman
    
    This MVP demonstration enhances Ultrahuman's existing glucose monitoring capabilities by adding an actionable layer
    that helps users optimize their meal sequences for better metabolic health.
    
    The food database contains detailed nutritional information and glycemic index values for a wide range of foods,
    allowing for practical meal planning and personalized recommendations.
    """)
    
    # Display key findings
    st.subheader("Key Research Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Impact of Food Order
        
        - **29-37% reduction** in glucose AUC when vegetables and protein are consumed before carbohydrates
        
        - **≈ 73% decrease** in insulin AUC when protein and vegetables precede carbohydrates
        
        - **Improved satiety** when fiber-rich foods are consumed first
        
        - **Slower gastric emptying** when fats and proteins precede carbohydrates
        """)
    
    with col2:
        st.markdown("""
        ### Clinical Applications
        
        - **Diabetes management:** Food sequencing can significantly improve glycemic control
        
        - **Weight management:** Reduced insulin spikes may decrease fat storage
        
        - **Improved energy:** Stabilized glucose levels prevent energy crashes
        
        - **Cardiovascular health:** Reduced glycemic variability improves heart health
        """)
    
    # References
    st.subheader("References")
    
    st.markdown("""
    1. Shukla AP, et al. "Food Order Has a Significant Impact on Postprandial Glucose and Insulin Levels." *Diabetes Care*. 2015;38(7):e98-e99.
    
    2. Trico D, et al. "Manipulating the sequence of food ingestion improves glycemic control in type 2 diabetic patients under free-living conditions." *Nutrition & Diabetes*. 2016;6:e226.
    
    3. Kuwata H, et al. "Meal sequence and glucose excursion, gastric emptying and incretin secretion in type 2 diabetes: a randomised, controlled crossover, exploratory trial." *Diabetologia*. 2016;59:453-461.
    
    4. Nishino K, et al. "The effect of the order of meal components on the blood glucose level. The glucose concentrations lowering effect of vegetables." *Jpn J Nutr Diet*. 2018;76:19-25.
    
    5. Imai S, et al. "Eating vegetables before carbohydrates improves postprandial glucose excursions." *Diabet Med*. 2013;30:370-372.
    """)

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:12px;">
    Ultrahuman Metabolic Score MVP Demo<br>
    Created for Ultrahuman Internship Application
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions for running the app if this is the first time
if 'first_run' not in st.session_state:
    st.session_state.first_run = True
    st.sidebar.title("Getting Started")
    st.sidebar.info("""
    👋 Welcome to the Ultrahuman Metabolic Score tool!
    
    To use this tool:
    1. Click on the **Food Sequence Impact** tab to learn about food sequencing effects
    2. Try the **Meal Builder** to create and optimize your meals
    3. Check the **About** tab to learn more about the science behind this tool
    """)

# Sidebar configuration
st.sidebar.title("Ultrahuman Metabolic Score")
st.sidebar.image("https://s3-recruiting.cdn.greenhouse.io/external_greenhouse_job_boards/logos/400/724/400/original/ULTRAHUMAN-LOGO-FINAL-01_(1).png?1666788357", width=200)

st.sidebar.header("About This Tool")
st.sidebar.markdown("""
This tool demonstrates how food sequencing affects glucose response and metabolic health. It's built based on clinical research and CGM data analysis.
""")

st.sidebar.header("Key Benefits")
st.sidebar.markdown("""
- **Lower glucose spikes** by 30-50%
- **Improved energy levels** throughout the day
- **Better weight management**
- **Reduced cravings** for sweets and snacks
- **Enhanced cognitive function**
""")

st.sidebar.header("How It Works")
st.sidebar.markdown("""
The tool analyzes:
- Food composition (carbs, protein, fat, fiber)
- Glycemic index values
- Food sequence optimization
- Predicted glucose response

It then calculates a Metabolic Score (1-100) where higher scores indicate better metabolic health.
""")

# Debug information (hidden in production)
debug_mode = False
if debug_mode:
    st.sidebar.header("Debug Information")
    st.sidebar.write(f"Food DB Rows: {len(food_db)}")
    st.sidebar.write(f"Food Types: {food_db['food_type'].unique()}")
    
    if st.sidebar.button("Show Sample Data"):
        st.sidebar.dataframe(food_db.head(5))
