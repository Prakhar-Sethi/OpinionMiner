"""
Streamlit UI for OpinionMiner.
Interactive personalized contrarian content feed.
"""

import streamlit as st
import sys
import os
import uuid
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.db_manager import DatabaseManager
from src.recommendation.feed_generator import FeedGenerator
from src.recommendation.bandit import ThompsonSamplingBandit

# Page configuration
st.set_page_config(
    page_title="OpinionMiner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .opinion-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .topic-tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 5px 5px 0;
    }
    .quality-dots {
        color: #ffd700;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_id = None
    st.session_state.current_feed = []
    st.session_state.current_index = 0
    st.session_state.start_time = None
    st.session_state.interactions_count = 0
    st.session_state.page = 'onboarding'

# Initialize database and feed generator
@st.cache_resource
def get_db_and_generator():
    """Get cached database and feed generator instances."""
    db = DatabaseManager()
    generator = FeedGenerator()
    return db, generator

db, generator = get_db_and_generator()

# Available topics
TOPICS = [
    'technology',
    'politics',
    'philosophy',
    'science',
    'culture',
    'economics',
    'education',
    'health',
    'environment',
    'sports',
    'entertainment',
    'business',
]

# Topic colors
TOPIC_COLORS = {
    'technology': '#4285f4',
    'politics': '#ea4335',
    'philosophy': '#9b59b6',
    'science': '#34a853',
    'culture': '#ff6d00',
    'economics': '#fbbc04',
    'education': '#46bdc6',
    'health': '#ff5252',
    'environment': '#00e676',
    'sports': '#ff9800',
    'entertainment': '#e91e63',
    'business': '#607d8b',
    'other': '#9e9e9e'
}


def render_onboarding():
    """Render onboarding page for new users."""
    st.title("🧠 Welcome to OpinionMiner")
    st.markdown("### Your Personalized Contrarian Content Feed")

    st.markdown("""
    OpinionMiner uses AI and multi-armed bandit learning to personalize your content feed in real-time.
    The more you interact, the better it understands your preferences!
    """)

    st.markdown("---")
    st.subheader("Select Your Interests")
    st.markdown("Choose at least 3 topics you're interested in:")

    # Topic selection
    selected_topics = []
    cols = st.columns(4)

    for i, topic in enumerate(TOPICS):
        with cols[i % 4]:
            if st.checkbox(topic.title(), key=f"topic_{topic}"):
                selected_topics.append(topic)

    st.markdown("---")

    # Validation and submission
    if len(selected_topics) < 3:
        st.warning("Please select at least 3 topics to continue.")
    else:
        st.success(f"Selected {len(selected_topics)} topics: {', '.join(selected_topics)}")

        if st.button("Get Started", type="primary"):
            # Create user profile
            user_id = str(uuid.uuid4())
            db.create_user_profile(user_id, selected_topics)

            # Update session state
            st.session_state.user_id = user_id
            st.session_state.initialized = True
            st.session_state.page = 'feed'

            st.success("Profile created! Redirecting to your feed...")
            time.sleep(1)
            st.rerun()


def render_opinion_card(opinion: dict, index: int):
    """Render a single opinion card."""
    # Quality dots
    quality_score = opinion.get('quality_score', 5.0)
    num_dots = min(5, int(quality_score / 2))
    quality_dots = "★" * num_dots + "☆" * (5 - num_dots)

    # Topic tag color
    topic = opinion.get('topic', 'other')
    topic_color = TOPIC_COLORS.get(topic, '#9e9e9e')

    st.markdown(f"""
    <div class="opinion-card">
        <h3>{opinion['title']}</h3>
        <div>
            <span class="topic-tag" style="background-color: {topic_color}; color: white;">
                {topic.upper()}
            </span>
            <span class="quality-dots">{quality_dots}</span>
        </div>
        <p style="color: #666; font-size: 14px;">
            by {opinion.get('author', 'Unknown')} • {opinion.get('source', 'Unknown')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Snippet
    content = opinion.get('content', '')
    snippet = content[:300] + "..." if len(content) > 300 else content
    st.markdown(snippet)


def render_feed():
    """Render the main feed page."""
    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Navigation")
        page = st.radio("Go to:", ["Feed", "Analytics", "Settings"], key="nav_radio")

        if page != "Feed":
            st.session_state.page = page.lower()
            st.rerun()

        st.markdown("---")
        st.markdown("### Stats")
        stats = db.get_interaction_stats(st.session_state.user_id)
        st.metric("Total Viewed", stats.get('total', 0))
        st.metric("Clicked", stats.get('clicks', 0))
        if stats.get('total', 0) > 0:
            st.metric("CTR", f"{stats.get('ctr', 0)*100:.1f}%")

    # Main feed
    st.title("🧠 OpinionMiner Feed")

    # Generate new feed if needed
    if (not st.session_state.current_feed or
        st.session_state.current_index >= len(st.session_state.current_feed) or
        st.session_state.interactions_count % 10 == 0 and st.session_state.interactions_count > 0):

        with st.spinner("Generating personalized feed..."):
            st.session_state.current_feed = generator.generate_feed(
                st.session_state.user_id,
                num_opinions=10
            )
            st.session_state.current_index = 0

    # Check if feed is empty
    if not st.session_state.current_feed:
        st.warning("No opinions available yet. Please run data collection first.")
        st.markdown("Run: `python -m src.data_collection.orchestrator`")
        return

    # Get current opinion
    if st.session_state.current_index < len(st.session_state.current_feed):
        opinion = st.session_state.current_feed[st.session_state.current_index]

        # Render opinion card
        render_opinion_card(opinion, st.session_state.current_index)

        # Interaction buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("👎 Skip", use_container_width=True):
                handle_skip(opinion)

        with col2:
            if st.button("📖 Read Full", use_container_width=True):
                handle_read(opinion)

        with col3:
            if st.button("👍 Next", use_container_width=True):
                handle_next(opinion)

        # Show full content if reading
        if f"reading_{opinion['id']}" in st.session_state and st.session_state[f"reading_{opinion['id']}"]:
            with st.expander("Full Content", expanded=True):
                st.markdown(opinion['content'])
                st.markdown(f"[View Source]({opinion['source_url']})")

                if st.button("Close"):
                    st.session_state[f"reading_{opinion['id']}"] = False
                    st.rerun()

    else:
        st.success("You've reached the end of your feed! Generating new opinions...")
        st.session_state.current_index = 0
        st.rerun()


def handle_skip(opinion: dict):
    """Handle skip interaction."""
    # Log interaction
    db.save_interaction(st.session_state.user_id, opinion['id'], 'skip')

    # Update bandit
    bandit = ThompsonSamplingBandit()
    bandit.update(st.session_state.user_id, opinion['topic'], reward=0)

    # Increment shown counter
    db.increment_opinion_shown(opinion['id'])

    # Move to next
    st.session_state.current_index += 1
    st.session_state.interactions_count += 1
    st.rerun()


def handle_read(opinion: dict):
    """Handle read interaction."""
    # Start tracking time
    st.session_state.start_time = time.time()

    # Show full content
    st.session_state[f"reading_{opinion['id']}"] = True

    # Log interaction
    db.save_interaction(st.session_state.user_id, opinion['id'], 'click')

    # Update bandit with positive reward
    bandit = ThompsonSamplingBandit()
    bandit.update(st.session_state.user_id, opinion['topic'], reward=1)

    # Increment shown counter
    db.increment_opinion_shown(opinion['id'])

    st.session_state.interactions_count += 1
    st.rerun()


def handle_next(opinion: dict):
    """Handle next interaction."""
    # Log interaction as viewed but not clicked
    db.save_interaction(st.session_state.user_id, opinion['id'], 'skip')

    # Update bandit
    bandit = ThompsonSamplingBandit()
    bandit.update(st.session_state.user_id, opinion['topic'], reward=0)

    # Increment shown counter
    db.increment_opinion_shown(opinion['id'])

    # Move to next
    st.session_state.current_index += 1
    st.session_state.interactions_count += 1
    st.rerun()


def render_analytics():
    """Render analytics page."""
    with st.sidebar:
        st.title("📊 Navigation")
        if st.button("← Back to Feed"):
            st.session_state.page = 'feed'
            st.rerun()

    st.title("📊 Analytics Dashboard")

    # Get bandit stats
    bandit = ThompsonSamplingBandit()
    arm_stats = bandit.get_arm_stats(st.session_state.user_id)

    if not arm_stats:
        st.warning("No interaction data yet. Start using the feed to see analytics!")
        return

    # Overall stats
    st.subheader("Overall Statistics")
    stats = db.get_interaction_stats(st.session_state.user_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interactions", stats.get('total', 0))
    with col2:
        st.metric("Clicks", stats.get('clicks', 0))
    with col3:
        st.metric("CTR", f"{stats.get('ctr', 0)*100:.1f}%")

    st.markdown("---")

    # Bandit arm stats
    st.subheader("Topic Performance (Bandit Arms)")

    # Create DataFrame
    df = pd.DataFrame(arm_stats)

    # Bar chart of estimated CTR
    fig = px.bar(
        df,
        x='topic',
        y='estimated_ctr',
        title='Estimated Click-Through Rate by Topic',
        labels={'estimated_ctr': 'Estimated CTR', 'topic': 'Topic'},
        color='estimated_ctr',
        color_continuous_scale='Blues'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Detailed Arm Statistics")
    display_df = df[['topic', 'alpha', 'beta', 'total_pulls', 'estimated_ctr']]
    display_df['estimated_ctr'] = display_df['estimated_ctr'].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(display_df, use_container_width=True)

    # Beta distributions visualization
    st.subheader("Confidence in Topic Preferences")
    st.markdown("Higher alpha/beta values = more confident in the estimate")

    fig2 = go.Figure()

    for stat in arm_stats[:6]:  # Top 6 topics
        topic = stat['topic']
        alpha = stat['alpha']
        beta = stat['beta']

        # Generate beta distribution curve
        import numpy as np
        from scipy import stats

        x = np.linspace(0, 1, 100)
        y = stats.beta.pdf(x, alpha, beta)

        fig2.add_trace(go.Scatter(
            x=x,
            y=y,
            name=topic,
            mode='lines'
        ))

    fig2.update_layout(
        title='Beta Distributions for Topic CTR',
        xaxis_title='Click-Through Rate',
        yaxis_title='Probability Density',
        hovermode='x unified'
    )

    st.plotly_chart(fig2, use_container_width=True)


def render_settings():
    """Render settings page."""
    with st.sidebar:
        st.title("📊 Navigation")
        if st.button("← Back to Feed"):
            st.session_state.page = 'feed'
            st.rerun()

    st.title("⚙️ Settings")

    st.subheader("User Profile")
    user_profile = db.get_user_profile(st.session_state.user_id)

    if user_profile:
        st.markdown(f"**User ID:** `{st.session_state.user_id}`")
        st.markdown(f"**Selected Topics:** {', '.join(user_profile['selected_topics'])}")
        st.markdown(f"**Created:** {user_profile['created_at']}")

    st.markdown("---")

    st.subheader("Preferences")

    # Exploration rate
    st.markdown("**Exploration Rate**")
    st.markdown("Higher values = more random exploration (default: 10%)")
    exploration_rate = st.slider(
        "Exploration Rate",
        min_value=0.0,
        max_value=0.3,
        value=0.1,
        step=0.05,
        format="%.2f",
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Reset recommendations
    st.subheader("Danger Zone")

    if st.button("🔄 Reset Recommendations", type="secondary"):
        if st.checkbox("Are you sure? This will reset your learning progress."):
            bandit = ThompsonSamplingBandit()
            bandit.reset_user_arms(st.session_state.user_id, user_profile['selected_topics'])
            st.success("Recommendations reset successfully!")
            time.sleep(1)
            st.rerun()


# Main app logic
def main():
    """Main application entry point."""
    # Check if user is initialized
    if not st.session_state.initialized:
        render_onboarding()
    else:
        # Route to appropriate page
        if st.session_state.page == 'feed':
            render_feed()
        elif st.session_state.page == 'analytics':
            render_analytics()
        elif st.session_state.page == 'settings':
            render_settings()


if __name__ == "__main__":
    main()
