from transformers import pipeline

# Question Answering
class QuestionAnswering:
    def __init__(self):
        self.qa_pipeline = pipeline('question-answering')

    def get_answer(self, question, context):
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']
