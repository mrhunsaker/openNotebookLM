import os
import asyncio
from nicegui import ui, app
from rag_pipeline import process_document_folder, chat_with_data, initialize_rag_pipeline, get_or_create_vector_store_for_notebook

# Load environment variables (for local testing outside of podman-compose)
from dotenv import load_dotenv
load_dotenv()

# Configuration from environment variables
APP_PORT = int(os.getenv("APP_PORT", 8000))
SOURCE_DIR = os.getenv("SOURCE_DIR", "/app/sources")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
DOCLING_API_URL = os.getenv("DOCLING_API_URL", "http://docling:8001")
MARKER_API_URL = os.getenv("MARKER_API_URL", "http://marker_ocr:8003")
GRANITE_MODEL_NAME = os.getenv("GRANITE_MODEL_NAME", "ibm/granite-3b-code-instruct")

# Global RAG components (LLM and Embedding model are shared)
rag_shared_components = {}
# Active notebook's vector store (changes based on selection)
active_vector_store = None
active_notebook_name = None

# --- Accessibility Enhancements ---
# Define a color palette that meets WCAG AA standards.
# These are just examples. You MUST verify these with a contrast checker.
# primary: a prominent brand color
# secondary: an accent color
# accent: another accent
# dark: for dark backgrounds
# positive, negative, info, warning: for status messages
ui.colors(primary='#1976D2', secondary='#9C27B0', accent='#00B0FF',
          dark='#1D1D1D', positive='#21BA45', negative='#C10015',
          info='#31CCEC', warning='#F2C037')

# Function to set accessible properties for buttons/inputs if needed
def set_accessible_button_props(button_element: ui.button, label: str):
    """Sets common accessibility properties for a button."""
    # Quasar buttons are generally accessible, but adding an explicit
    # aria-label for clarity can sometimes help, especially if text is omitted.
    button_element.props(f'aria-label="{label}"')

def set_accessible_input_props(input_element: ui.input, label: str):
    """Sets common accessibility properties for an input."""
    # NiceGUI ui.input typically handles `label` attribute which maps to aria-label/labelledby.
    # No extra props often needed here unless custom styling hides the label.
    pass # Currently, NiceGUI's `label` handles this well.


@ui.page('/')
def main_page():
    ui.add_head_html('''
        <style>
            /* Visually hidden for screen readers */
            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                margin: -1px;
                padding: 0;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                border: 0;
                white-space: nowrap; /* Keep the text on a single line */
            }
        </style>
    ''')

    # --- Global Live Region for Status Messages ---
    # This element will be used for visually hidden updates that screen readers announce.
    # It must be present from the start.
    status_aria_live_region = ui.element('div').props('aria-live="polite" aria-atomic="true" class="sr-only"').style('position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0;')

    global active_notebook_name, active_vector_store

    async def send_message():
        query = user_input.value
        if not query:
            ui.notify("Please type your question.", type='warning')
            status_aria_live_region.set_text("Please type your question.")
            return

        if not active_notebook_name or not active_vector_store:
            ui.notify("Please select a notebook before chatting.", type='warning')
            status_aria_live_region.set_text("Please select a notebook before chatting.")
            return

        # Append user message
        with chat_history:
            user_message_row = ui.row().classes('w-full items-start').on('click', lambda: ui.run_javascript(f'navigator.clipboard.writeText("{query.replace("\"", "\\\"")}")', respond=False))
            with user_message_row:
                ui.avatar('person').classes('mr-2').props('aria-hidden="true"') # Decorative, hide from SR
                ui.markdown(f"**You:** {query}").classes('p-2 bg-blue-100 rounded-lg max-w-[80%]')
                # Add a visually hidden span for screen reader to announce message submission
                ui.element('span').classes('sr-only').set_text(f"You said: {query}")

        user_input.set_value('')
        status_aria_live_region.set_text("Message sent. Awaiting AI response.")

        if not rag_shared_components or not active_vector_store:
            ui.notify("RAG components not initialized or notebook not selected. Please wait or check logs.", type='negative')
            status_aria_live_region.set_text("RAG system not ready. Please try again later.")
            with chat_history:
                ui.row().classes('w-full items-start').add(
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"'),
                    ui.markdown("**AI:** RAG system is not ready. Please try again later.").classes('p-2 bg-gray-200 rounded-lg max-w-[80%]'),
                )
            return

        # Get response from RAG pipeline
        try:
            response_text, citations = await chat_with_data(
                query,
                active_vector_store,
                rag_shared_components['llm']
            )

            full_response = response_text
            if citations:
                full_response += "\n\n**Sources:**\n"
                for i, citation in enumerate(citations):
                    full_response += f"{i+1}. {citation}\n"

            with chat_history:
                ai_message_row = ui.row().classes('w-full items-start')
                with ai_message_row:
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"') # Decorative, hide from SR
                    ai_markdown_element = ui.markdown(f"**AI:** {full_response}").classes('p-2 bg-gray-200 rounded-lg max-w-[80%]')
                    # For screen readers, ensure the entire response is available and announced
                    # The chat_history's aria-live="polite" should handle this.
                    # If the response is very long, consider summarizing or breaking it down.
            status_aria_live_region.set_text("AI response received.")
        except Exception as e:
            with chat_history:
                ui.row().classes('w-full items-start').add(
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"'),
                    ui.markdown(f"**AI:** An error occurred during chat: {e}").classes('p-2 bg-red-200 rounded-lg max-w-[80%]'),
                )
            ui.notify(f"Error during chat: {e}", type='negative')
            status_aria_live_region.set_text(f"Error during chat: {e}.")

    # Using a unique ID for the select element, which is then linked by the input
    notebook_select_id = 'notebook-select-input'
    notebook_selector = ui.select(
        options=[],
        with_input=True,
        label="Select or Create Notebook",
        value=active_notebook_name, # Initialize with current active notebook if any
        # Make sure the select element itself has a proper label. ui.select(label=...) handles this.
        # Add role=combobox etc. if custom styles break default behavior. NiceGUI handles this.
    ).props(f'clearable aria-labelledby="{notebook_select_id}"').classes('w-full').on('update:model-value', lambda e: ui.run_javascript('location.reload()')) # Reload for now, will improve

    current_notebook_folder_input = ui.input(
        label="Notebook Folder Path (relative to ~/downloads/openLM/sources)",
        placeholder="e.g., my_research_notes",
        # Validation for input:
        validation={'Please enter a folder name.': lambda value: bool(value and value.strip())}
    ).props('clearable').classes('w-full')
    set_accessible_input_props(current_notebook_folder_input, "Notebook Folder Path")


    processing_status = ui.label("Ready.").classes('ml-4 text-sm text-gray-600')
    # Make the processing_status an ARIA live region for screen readers.
    # Its content will be spoken when it changes.
    processing_status.props('aria-live="polite" aria-atomic="true"')

    # Chat History - Make it an ARIA live region if new messages are frequently added
    # without user interaction (e.g., AI response appears).
    chat_history = ui.column().classes('overflow-auto h-96 border border-gray-300 rounded-lg p-4 mb-4 bg-gray-50').props('aria-live="polite" aria-atomic="false"')
    # When new messages are added, screen readers should announce them.
    # Setting aria-atomic="false" means only the added content is announced, not the whole region.

    user_input = ui.input(label="Your question", placeholder="Ask about your documents...")
    user_input.props('clearable').on('keydown.enter', send_message).classes('w-full')
    set_accessible_input_props(user_input, "Your question")


    async def update_notebook_options():
        """Populates the notebook selector with existing folders."""
        notebook_folders = []
        if os.path.exists(SOURCE_DIR):
            notebook_folders = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
        notebook_selector.options = notebook_folders
        if notebook_folders:
            # Set default if none selected or invalid previous selection
            if active_notebook_name not in notebook_folders:
                active_notebook_name = notebook_folders[0]
            notebook_selector.set_value(active_notebook_name)
            await select_notebook_for_chat(active_notebook_name)
        else:
            notebook_selector.set_value(None)
            active_notebook_name = None
        status_aria_live_region.set_text("Notebook options updated.")


    async def initialize_app_components():
        global rag_shared_components
        processing_status.set_text("Initializing shared RAG components (LLM, Embeddings)...")
        status_aria_live_region.set_text("Initializing application components.")
        try:
            rag_shared_components = await initialize_rag_pipeline(
                granite_model_name=GRANITE_MODEL_NAME,
                chroma_persist_dir=CHROMA_PERSIST_DIR
            )
            processing_status.set_text("Shared RAG components initialized.")
            ui.notify("RAG components initialized successfully!", type='positive')
            status_aria_live_region.set_text("RAG components initialized successfully.")
            await update_notebook_options()
        except Exception as e:
            processing_status.set_text(f"Error initializing RAG: {e}")
            ui.notify(f"Error initializing RAG: {e}", type='negative')
            status_aria_live_region.set_text(f"Error initializing RAG: {e}")


    async def select_notebook_for_chat(notebook_name: str):
        global active_notebook_name, active_vector_store
        if not notebook_name:
            active_notebook_name = None
            active_vector_store = None
            ui.notify("No notebook selected.", type='warning')
            status_aria_live_region.set_text("No notebook selected.")
            return

        active_notebook_name = notebook_name
        processing_status.set_text(f"Switching to notebook: {active_notebook_name}")
        status_aria_live_region.set_text(f"Switching to notebook: {active_notebook_name}")
        try:
            active_vector_store = await get_or_create_vector_store_for_notebook(
                notebook_name,
                rag_shared_components['embedding_model'],
                CHROMA_PERSIST_DIR
            )
            processing_status.set_text(f"Ready to chat with '{active_notebook_name}'.")
            ui.notify(f"Switched to notebook '{active_notebook_name}'.", type='info')
            status_aria_live_region.set_text(f"Switched to notebook '{active_notebook_name}'. Chat history cleared.")
            chat_history.clear()
        except Exception as e:
            processing_status.set_text(f"Error selecting notebook: {e}")
            ui.notify(f"Error selecting notebook: {e}", type='negative')
            status_aria_live_region.set_text(f"Error selecting notebook: {e}")


    async def process_selected_notebook():
        # Validate input field
        if not current_notebook_folder_input.value or not current_notebook_folder_input.value.strip():
            current_notebook_folder_input.props('error', True)
            current_notebook_folder_input.props('error-message', 'Please enter a folder name.')
            ui.notify("Please enter a notebook folder name.", type='warning')
            status_aria_live_region.set_text("Validation error: Please enter a notebook folder name.")
            return

        folder_name = current_notebook_folder_input.value
        full_source_path = os.path.join(SOURCE_DIR, folder_name)
        if not os.path.isdir(full_source_path):
            ui.notify(f"Folder '{full_source_path}' does not exist. Please create it first.", type='negative')
            status_aria_live_region.set_text(f"Error: Folder {folder_name} does not exist.")
            return

        processing_status.set_text(f"Processing documents in '{folder_name}'...")
        ui.notify(f"Starting document processing for '{folder_name}'...", type='info')
        status_aria_live_region.set_text(f"Starting document processing for {folder_name}.")

        try:
            await select_notebook_for_chat(folder_name)
            if not active_vector_store:
                raise Exception("Failed to get/create vector store for notebook.")

            await process_document_folder(
                full_source_path,
                active_vector_store,
                rag_shared_components['embedding_model'],
                docling_api_url=DOCLING_API_URL,
                marker_api_url=MARKER_API_URL
            )
            processing_status.set_text(f"Finished processing documents in '{folder_name}'. Ready to chat.")
            ui.notify(f"Documents in '{folder_name}' processed successfully!", type='positive')
            status_aria_live_region.set_text(f"Documents in {folder_name} processed successfully.")
            await update_notebook_options()
        except Exception as e:
            processing_status.set_text(f"Error processing documents: {e}")
            ui.notify(f"Error processing documents: {e}", type='negative')
            status_aria_live_region.set_text(f"Error processing documents for {folder_name}: {e}.")


    async def send_message():
        query = user_input.value
        if not query:
            ui.notify("Please type your question.", type='warning')
            status_aria_live_region.set_text("Please type your question.")
            return

        if not active_notebook_name or not active_vector_store:
            ui.notify("Please select a notebook before chatting.", type='warning')
            status_aria_live_region.set_text("Please select a notebook before chatting.")
            return

        # Append user message
        with chat_history:
            user_message_row = ui.row().classes('w-full items-start').on('click', lambda: ui.run_javascript(f'navigator.clipboard.writeText("{query.replace("\"", "\\\"")}")', respond=False))
            with user_message_row:
                ui.avatar('person').classes('mr-2').props('aria-hidden="true"') # Decorative, hide from SR
                ui.markdown(f"**You:** {query}").classes('p-2 bg-blue-100 rounded-lg max-w-[80%]')
                # Add a visually hidden span for screen reader to announce message submission
                ui.element('span').classes('sr-only').set_text(f"You said: {query}")

        user_input.set_value('')
        status_aria_live_region.set_text("Message sent. Awaiting AI response.")


        if not rag_shared_components or not active_vector_store:
            ui.notify("RAG components not initialized or notebook not selected. Please wait or check logs.", type='negative')
            status_aria_live_region.set_text("RAG system not ready. Please try again later.")
            with chat_history:
                ui.row().classes('w-full items-start').add(
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"'),
                    ui.markdown("**AI:** RAG system is not ready. Please try again later.").classes('p-2 bg-gray-200 rounded-lg max-w-[80%]'),
                )
            return

        # Get response from RAG pipeline
        try:
            response_text, citations = await chat_with_data(
                query,
                active_vector_store,
                rag_shared_components['llm']
            )

            full_response = response_text
            if citations:
                full_response += "\n\n**Sources:**\n"
                for i, citation in enumerate(citations):
                    full_response += f"{i+1}. {citation}\n"

            with chat_history:
                ai_message_row = ui.row().classes('w-full items-start')
                with ai_message_row:
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"') # Decorative, hide from SR
                    ai_markdown_element = ui.markdown(f"**AI:** {full_response}").classes('p-2 bg-gray-200 rounded-lg max-w-[80%]')
                    # For screen readers, ensure the entire response is available and announced
                    # The chat_history's aria-live="polite" should handle this.
                    # If the response is very long, consider summarizing or breaking it down.
            status_aria_live_region.set_text("AI response received.")
        except Exception as e:
            with chat_history:
                ui.row().classes('w-full items-start').add(
                    ui.avatar('robot_2').classes('mr-2').props('aria-hidden="true"'),
                    ui.markdown(f"**AI:** An error occurred during chat: {e}").classes('p-2 bg-red-200 rounded-lg max-w-[80%]'),
                )
            ui.notify(f"Error during chat: {e}", type='negative')
            status_aria_live_region.set_text(f"Error during chat: {e}.")


    # --- UI Layout ---
    # Use semantic HTML elements for headings (h1, h2, etc.)
    # Ensure proper tab order (elements rendered in order will generally follow tab order)

    with ui.header().classes('items-center justify-between bg-primary text-white'): # Use primary color
        ui.label('Local OpenLM RAG Notebook').classes('text-2xl font-bold').props('role="heading" aria-level="1"') # Explicit heading role

    with ui.card().classes('w-full max-w-4xl mx-auto p-4 mb-4 shadow-md'):
        ui.label('Notebook Management').classes('text-xl font-semibold mb-2').props('role="heading" aria-level="2"') # Explicit heading role

        # Grouping for accessibility with aria-labelledby
        with ui.column().classes('w-full'):
            ui.label('Select or define a notebook folder:').props('id="notebook-select-label"').classes('text-lg')
            notebook_selector.props('aria-labelledby="notebook-select-label"') # Link the label to the select

            # Bind select value to input for visual consistency, keyboard accessibility handled by select
            notebook_selector.bind_value_to(current_notebook_folder_input, 'value')

            ui.button('Select Notebook', on_click=lambda: select_notebook_for_chat(notebook_selector.value)).classes('mt-2 bg-secondary text-white')
            set_accessible_button_props(ui.get_template()._props[-1], "Select Notebook to chat with") # Access the last created button


        current_notebook_folder_input.classes('mt-4') # Input to define new notebook folder path

        # Button for processing documents
        process_button = ui.button('Process Documents in this Folder', on_click=process_selected_notebook).classes('mt-2 bg-positive text-white')
        set_accessible_button_props(process_button, "Process documents in the selected or new notebook folder")

        processing_status # Status label already defined with aria-live


    ui.separator().classes('my-6') # Add some visual separation

    with ui.card().classes('w-full max-w-4xl mx-auto p-4 shadow-md'):
        # Dynamic heading for chat context
        chat_heading = ui.label('Chat with: No Notebook Selected').classes('text-xl font-semibold mb-4').props('role="heading" aria-level="2"')
        chat_heading.bind_text_from(globals(), 'active_notebook_name', lambda name: f'Chat with: {name if name else "No Notebook Selected"}')


        chat_history # Chat history area already defined with aria-live

        # Chat input and send button
        with ui.row().classes('w-full items-center gap-2 mt-4'):
            user_input.classes('flex-grow') # Make input take available space
            send_button = ui.button('Send', on_click=send_message).classes('bg-primary text-white')
            set_accessible_button_props(send_button, "Send your question to the AI")


    # Run initialization on app startup
    app.on_startup(initialize_app_components)

# Run the NiceGUI app
ui.run(port=APP_PORT, title="Local OpenLM RAG Notebook", dark=False)
