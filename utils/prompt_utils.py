import os
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from utils.logging_utils import logger
from langchain import hub


class PromptUtils:
    def format_docs_chunks(self, docs):
        logger.info(f"**[Debug] Retrieved {len(docs)} chunks.**")
        normalized_docs = []
        for i, doc in enumerate(docs):
            # Extract metadata if available
            # Assuming each doc has metadata with 'page' and 'source'
            page_num = doc.metadata.get("page") + 1
            source = doc.metadata.get("source", "document")
            file_name = (
                os.path.basename(source) if isinstance(source, str) else "unknown"
            )

            extract = {
                "index": i + 1,
                "page": page_num,
                "source": file_name,
                "content": doc.page_content.strip(),
            }
            logger.info(
                f"""
            ([Reference-{extract["index"]}] Page {extract["page"]} - Source: {extract["source"]})
            {extract["content"]}"""
            )

            normalized_docs.append(extract)

        # Join all document contents into a single string
        # This is useful for displaying or processing the content further
        result = "\n\n".join(doc["content"] for doc in normalized_docs)
        return result

    def format_conversation_history(self, histories):
        formatted_history = ""
        for msg in histories:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        return formatted_history.strip()

    def load_prompt_template_from_hub(self, name):
        return hub.pull(name)

    def load_prompt_template_from_file(self, filename):
        """Load a prompt from a text file."""
        template = None
        with open(f"./prompt_templates/{filename}", "r", encoding="utf-8") as file:
            template = file.read()
        return ChatPromptTemplate.from_template(template)


# Create a singleton instance
promptManager = PromptUtils()
