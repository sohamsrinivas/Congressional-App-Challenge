import json

import pandas as pd
import streamlit as st

from pipeline import load_artifacts, run_batch, run_pipeline

st.set_page_config(page_title="TicketIQ", page_icon="🎟️", layout="wide")

# ---------------------------------------------------------------------------
# Load models once per server
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return load_artifacts()


art = get_artifacts()

SAMPLES = {
    "Locked out, angry": "I am unable to login to my account. This issue is extremely frustrating and I need urgent support, I have a deadline tomorrow.",
    "Double charge": "I was charged twice this month for my subscription. Please refund the second charge.",
    "Simple question": "Hi, how do I change the email address on my profile?",
    "Happy customer": "Thanks so much, the new update is great and the app is way faster now!",
    "Worried about security": "I got an email saying someone logged in from another country. Was my account hacked? I'm really worried.",
}

RISK_COLORS = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎟️ TicketIQ")
    st.caption("Reads a support message and tells a small team what to do about it first.")

    st.subheader("Customer history")
    st.caption("Optional. These feed the churn model so one bad message doesn't decide everything.")
    ticket_count = st.slider("Previous tickets from this customer", 0, 20, 1)
    usage = st.slider("Product usage (0 = inactive, 100 = daily)", 0, 100, 50)
    avg_sentiment = st.slider("Average tone of past tickets", -1.0, 1.0, 0.0, 0.1)
    context = {"ticket_count": ticket_count, "usage": usage, "avg_sentiment": avg_sentiment}

    st.subheader("Model status")
    for name, ok in art["status"].items():
        st.write(("✅ " if ok else "⚠️ ") + name + ("" if ok else " (rule-based fallback)"))

    with st.expander("How it works"):
        st.markdown(
            """
            1. **Classify** the ticket into a department and a priority.
            2. **Read the tone**: sentiment score plus a named emotion.
            3. **Tag the topic** so repeat problems get spotted.
            4. **Estimate churn risk** from tone plus customer history.
            5. **Score business risk** (0 to 100) and generate next steps.

            Every result shows which method produced it, so nothing is a black box.
            """
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Support ticket triage, in one click")
st.write(
    "Small teams, school help desks, and nonprofits get buried in messages and answer them in the order they arrive. "
    "TicketIQ reads each one and surfaces the ones that actually need a human right now."
)

tab_single, tab_batch = st.tabs(["Analyze one ticket", "Batch (CSV upload)"])

# ----- Single ticket ------------------------------------------------------
with tab_single:
    col_in, col_samples = st.columns([3, 1])
    with col_samples:
        st.markdown("**Try a sample**")
        for label, txt in SAMPLES.items():
            if st.button(label, use_container_width=True):
                st.session_state["ticket_text"] = txt
    with col_in:
        text = st.text_area(
            "Paste a support message",
            key="ticket_text",
            height=160,
            placeholder="e.g. My order never arrived and nobody is answering my emails...",
        )
        run = st.button("Analyze", type="primary", disabled=not text.strip())

    if run and text.strip():
        r = run_pipeline(text, art, context)

        st.divider()
        st.subheader("Summary")
        st.markdown(f"**{r['summary']}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Priority", r["priority"])
        m2.metric("Sentiment", r["sentiment"], f"{r['sentiment_score']:+.2f}")
        m3.metric("Churn risk", r["churn_risk"], f"{r['churn_probability']:.0%} probability", delta_color="inverse")
        color = RISK_COLORS.get(r["business_risk_level"], "gray")
        m4.metric("Business risk", f"{r['business_risk_score']:.0f} / 100")
        m4.markdown(f":{color}[**{r['business_risk_level']}**]")

        st.progress(min(int(r["business_risk_score"]), 100))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("What the model saw")
            st.table(pd.DataFrame(
                [
                    ["Category", r["category"], r["category_source"]],
                    ["Priority", r["priority"], "keyword rules"],
                    ["Emotion", r["emotion"], "keyword rules"],
                    ["Sentiment", f"{r['sentiment']} ({r['sentiment_score']:+.2f})", r["sentiment_source"]],
                    ["Topic", f"{r['topic_label']} ({r['topic_confidence']:.0%})", r["topic_source"]],
                    ["Churn", f"{r['churn_risk']} ({r['churn_probability']:.0%})", r["churn_source"]],
                ],
                columns=["Signal", "Result", "Method"],
            ).set_index("Signal"))
        with c2:
            st.subheader("Recommended next steps")
            for i, rec in enumerate(r["recommendations"], 1):
                st.markdown(f"{i}. {rec}")

        export = {k: v for k, v in r.items() if not k.endswith("_source")}
        st.download_button(
            "Download result as JSON",
            data=json.dumps(export, indent=2),
            file_name="ticketiq_result.json",
            mime="application/json",
        )

# ----- Batch --------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV with one ticket per row. The app will score every ticket and sort by business risk.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        text_col = st.selectbox("Which column holds the ticket text?", df_in.columns)
        if st.button("Score all tickets", type="primary"):
            with st.spinner("Scoring..."):
                out = run_batch(df_in[text_col].astype(str).tolist(), art, context)
            if out.empty:
                st.warning("No non-empty tickets found.")
            else:
                out = out.sort_values("business_risk_score", ascending=False)
                k1, k2, k3 = st.columns(3)
                k1.metric("Tickets scored", len(out))
                k2.metric("Critical or High risk", int(out["business_risk_level"].isin(["Critical", "High"]).sum()))
                k3.metric("High churn risk", int((out["churn_risk"] == "High").sum()))

                st.bar_chart(out["category"].value_counts())
                show_cols = ["business_risk_score", "business_risk_level", "priority", "category", "sentiment", "emotion", "topic_label", "churn_risk", "ticket_text"]
                st.dataframe(out[show_cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scored CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="ticketiq_scored.csv",
                    mime="text/csv",
                )
    else:
        demo = pd.DataFrame({"ticket": list(SAMPLES.values())})
        st.download_button("Need a test file? Download a sample CSV", demo.to_csv(index=False).encode("utf-8"), "sample_tickets.csv", "text/csv")
