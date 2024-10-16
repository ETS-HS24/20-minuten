from transformers import BartTokenizer, BartForConditionalGeneration

class BartTopicModeling:
    def __init__(self, model_name='facebook/bart-base'):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def generate_topics(self, text, num_topics=5, max_length=50):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            num_beams=num_topics,
            max_length=max_length,
            early_stopping=True,
            no_repeat_ngram_size=2  # Prevents repetition
        )
        topics = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        return topics

    def print_topics(self, topics):
        for i, topic in enumerate(topics):
            print(f"Topic {i+1}: {topic}")

if __name__ == "__main__":
    text = "Storm Sabine throws Zurich Airport's plan into disarray. A video from a reader-reporter shows the A380 having to abort the landing. The Singapore Airline A380 had to abort its approach to Kloten Airport. It worked the second time, as a video from a reader-reporter shows. It wasn't just the A380 giant that had to take off. As a reader reporter reports, a Swiss A330-300 was also hit. The plane came from Tel Aviv."
    bart_topic_model = BartTopicModeling()
    topics = bart_topic_model.generate_topics(text, num_topics=5, max_length=50)
    bart_topic_model.print_topics(topics)
