import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Phishing Email Detector", page_icon="🛡️")

st.title("🛡️ AI-Based Phishing Email Detector")
st.write("Paste an email below to check whether it may be phishing or legitimate.")

data = {
    "email": [
        "Your bank account is blocked. Click here to verify immediately.",
        "Congratulations! You won a free prize. Claim it now.",
        "Your account will be suspended. Update your password now.",
        "Urgent! Send your OTP to avoid account closure.",
        "Click this link to receive your free gift card.",
        "Your invoice is attached for your review.",
        "The meeting is scheduled for tomorrow at 10 AM.",
        "Thank you for your application. We will contact you soon.",
        "Please find the project report attached.",
        "Your appointment is confirmed for Monday.",
        "Verify your account now to avoid suspension.",
        "You have won a lottery. Send your bank details.",
        "Your password expires today. Click here to reset it.",
        "Claim your cash reward immediately.",
        "We detected suspicious activity. Login here now.",
        "The office will remain closed tomorrow.",
        "Please review the attached document.",
        "Your order has been shipped and is on its way.",
        "The class starts at 9 AM tomorrow.",
        "Thank you for attending the interview.",
        "You are selected for a cash prize. Click now.",
        "Your email account will be deleted unless you verify.",
        "Send your password to confirm your identity.",
        "Limited offer! Claim your free mobile today.",
        "Your package delivery is scheduled for tomorrow.",
        "The team meeting has been moved to Friday.",
        "Please complete the registration form.",
        "Your payment receipt is attached.",
        "We will share the updated schedule shortly.",
        "Your leave request has been approved."
    ],
    "label": [
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Legitimate", "Legitimate", "Legitimate", "Legitimate", "Legitimate",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Legitimate", "Legitimate", "Legitimate", "Legitimate", "Legitimate",
        "Phishing", "Phishing", "Phishing", "Phishing", "Phishing",
        "Legitimate", "Legitimate", "Legitimate", "Legitimate", "Legitimate"
    ]
}

df = pd.DataFrame(data)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(df["email"], df["label"])

email = st.text_area("Paste your email here:", height=180)

if st.button("Check Email"):
    if email.strip():
        prediction = model.predict([email])[0]
        probabilities = model.predict_proba([email])[0]
        confidence = max(probabilities) * 100

        if prediction == "Phishing":
            st.error("⚠️ This email may be PHISHING.")
        else:
            st.success("✅ This email appears LEGITIMATE.")

        st.metric("Model Confidence", f"{confidence:.2f}%")
        st.progress(float(confidence / 100))
        st.caption("Confidence is a model estimate, not a guarantee. Always verify suspicious links and requests.")
    else:
        st.warning("Please paste an email first.")

