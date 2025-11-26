"""Main Shiny application for RecycLens MVP."""
import base64
import os
from io import BytesIO
from typing import Optional, Dict, Any
from pathlib import Path

from shiny import App, ui, reactive, render
from shiny.types import FileInfo
from PIL import Image
import asyncio

from utils.api_client import analyze_vision, analyze_recyclability
from dotenv import load_dotenv

load_dotenv()

MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN", "")

# Analysis stages
STAGE_IDLE = "idle"
STAGE_ANALYZING_IMAGE = "analyzing-image"
STAGE_ANALYZING_RECYCLABILITY = "analyzing-recyclability"
STAGE_GEOCODING = "geocoding-facilities"
STAGE_COMPLETE = "complete"
STAGE_ERROR = "error"


def convert_image_to_base64(file_info: FileInfo) -> str:
    """Convert uploaded image to base64 data URL."""
    with open(file_info["datapath"], "rb") as f:
        image_data = f.read()
        base64_str = base64.b64encode(image_data).decode("utf-8")
        # Determine MIME type from file extension
        ext = Path(file_info["name"]).suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")
        return f"data:{mime_type};base64,{base64_str}"


def nav_bar():
    """Navigation bar component."""
    return ui.div(
        ui.div(
            ui.input_action_link(
                "nav_logo",
                ui.span("RecycLens", class_="text-xl font-light tracking-tight"),
                class_="nav-logo-link"
            ),
            ui.div(
                ui.input_action_link(
                    "nav_to_how_it_works",
                    "How it Works",
                    class_="text-sm text-gray-600 hover:text-gray-900 transition-colors"
                ),
                class_="flex items-center space-x-8"
            ),
            class_="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between"
        ),
        class_="bg-white/80 backdrop-blur-md border-b border-gray-100 sticky top-0 z-50"
    )


def hero_section():
    """Hero section with title and description."""
    return ui.div(
        ui.div(
            ui.h1(
                "Know What to Recycle,",
                ui.br(),
                "Instantly.",
                class_="text-6xl font-light tracking-tight text-gray-900 mb-4"
            ),
            ui.p(
                "Set your location, upload an item photo, add context, and get clear recycling guidance.",
                class_="text-xl text-gray-600 font-light max-w-2xl mx-auto"
            ),
            class_="text-center mb-12"
        ),
        class_="max-w-7xl mx-auto px-6 py-16"
    )


def input_form():
    """Input form with location, context, and image upload."""
    return ui.div(
        ui.div(
            # Location Input
            ui.div(
                ui.tags.label(
                    "Location ",
                    ui.span("*", class_="text-red-500"),
                    class_="text-sm font-medium text-gray-700 mb-2 block"
                ),
                ui.div(
                    ui.span("📍", class_="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 z-10"),
                    ui.input_text(
                        "location",
                        "",
                        placeholder="Enter your city or ZIP code"
                    ),
                    class_="relative"
                ),
                class_="form-group"
            ),
            
            # Context Input
            ui.div(
                ui.tags.label(
                    "Add additional context ",
                    ui.span("(optional)", class_="text-gray-400 text-xs font-normal"),
                    class_="text-sm font-medium text-gray-700 mb-2 block"
                ),
                ui.input_text_area(
                    "context",
                    "",
                    placeholder="e.g., Plastic container with food residue",
                    rows=3
                ),
                class_="form-group"
            ),
            
            # Image Upload
            ui.div(
                ui.tags.label(
                    "Item Photo ",
                    ui.span("*", class_="text-red-500"),
                    class_="text-sm font-medium text-gray-700 mb-2 block"
                ),
                ui.div(
                    # The invisible file input covering the container
                    ui.div(
                        ui.input_file(
                            "image",
                            label="",
                            accept="image/*",
                            multiple=False
                        ),
                        class_="file-upload-input-wrapper"
                    ),
                    # The visual content
                    ui.div(
                        ui.span("📷", class_="text-4xl mb-2 block"),
                        ui.p("Drop an image here or click to browse", class_="text-gray-600 mb-1"),
                        ui.p("Upload or Take a Photo", class_="text-sm text-gray-400"),
                        class_="file-upload-content"
                    ),
                    class_="file-upload-container"
                ),
                ui.div(
                    ui.output_image("image_preview"),
                    class_="mt-4 flex justify-center"
                ),
                ui.output_ui("remove_image_ui"),
                class_="mb-6"
            ),
            
            # Error Display
            ui.output_ui("error_display"),
            
            # Progress Indicator
            ui.output_ui("progress_indicator"),
            
            # CTA Button
            ui.div(
                ui.input_action_button(
                    "check_button",
                    "Check if it's Recyclable",
                    class_="w-full bg-gradient-to-r from-green-500 to-green-600 text-white py-4 rounded-full font-medium hover:from-green-600 hover:to-green-700 transition-all shadow-lg shadow-green-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                ),
                class_="mb-6"
            ),
            class_="bg-white rounded-3xl shadow-sm border border-gray-100 p-8 transition-all duration-700 ease-in-out"
        ),
        class_="max-w-2xl mx-auto"
    )


def facility_map_section():
    """Facility map and cards section."""
    return ui.div(
        ui.h2(
            "Nearby Recycling Facilities",
            class_="text-4xl font-light text-gray-900 mb-8 text-center"
        ),
        ui.div(
            ui.output_ui("facility_map"),
            ui.output_ui("facility_cards"),
            class_="bg-white rounded-3xl shadow-sm border border-gray-100 overflow-hidden"
        ),
        class_="mt-20"
    )


def how_it_works_page():
    """How it Works page component."""
    steps = [
        {
            "icon": "📷",
            "title": "Upload Your Item Photo",
            "description": "Take or upload a clear photo of the item you want to recycle or dispose of."
        },
        {
            "icon": "📍",
            "title": "Enter Your Location",
            "description": "Provide your city, state, or ZIP code so we can find nearby facilities."
        },
        {
            "icon": "📝",
            "title": "Add Context (Optional)",
            "description": "Describe any special conditions, like if the item is greasy, broken, or contaminated."
        },
        {
            "icon": "✨",
            "title": "Get Instant Analysis",
            "description": "Our AI analyzes the material, determines recyclability, and provides disposal instructions."
        },
        {
            "icon": "🗺️",
            "title": "Find Nearby Facilities",
            "description": "View recycling centers and disposal facilities on an interactive map with directions."
        }
    ]
    
    step_cards = [
        ui.div(
            ui.div(
                ui.div(
                    ui.span(
                        step["icon"],
                        class_="text-4xl"
                    ),
                    class_="w-16 h-16 bg-gradient-to-br from-green-400 to-green-600 rounded-2xl flex items-center justify-center"
                ),
                ui.div(
                    ui.div(
                        ui.span(
                            f"Step {i+1}",
                            class_="text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full"
                        ),
                        ui.h3(
                            step["title"],
                            class_="text-2xl font-light text-gray-900"
                        ),
                        class_="flex items-center gap-3 mb-2"
                    ),
                    ui.p(
                        step["description"],
                        class_="text-gray-600 text-lg"
                    ),
                    class_="flex-1"
                ),
                class_="bg-white rounded-3xl shadow-sm border border-gray-100 p-8 flex items-start gap-6"
            ),
            class_="mb-8"
        )
        for i, step in enumerate(steps)
    ]
    
    return ui.div(
        ui.div(
            ui.h1(
                "How ",
                ui.span("RecycLens", class_="bg-gradient-to-r from-green-500 to-green-600 bg-clip-text text-transparent"),
                " Works",
                class_="text-6xl font-light text-gray-900 mb-4 text-center"
            ),
            ui.p(
                "Get instant, AI-powered recycling guidance in just a few simple steps.",
                class_="text-xl text-gray-600 text-center max-w-2xl mx-auto mb-16"
            ),
            ui.div(*step_cards, class_="space-y-8 mb-16"),
            
            # Benefits Section
            ui.div(
                ui.h2(
                    "Why Use RecycLens?",
                    class_="text-3xl font-light text-gray-900 mb-6 text-center"
                ),
                ui.div(
                    ui.div(
                        ui.div("⚡", class_="text-4xl font-light text-green-600 mb-2"),
                        ui.h3("Instant Results", class_="text-xl font-medium text-gray-900 mb-2"),
                        ui.p(
                            "Get recycling guidance in seconds, not hours of research.",
                            class_="text-gray-600"
                        ),
                        class_="text-center"
                    ),
                    ui.div(
                        ui.div("🎯", class_="text-4xl font-light text-green-600 mb-2"),
                        ui.h3("Accurate Guidance", class_="text-xl font-medium text-gray-900 mb-2"),
                        ui.p(
                            "AI-powered analysis ensures you dispose of items correctly.",
                            class_="text-gray-600"
                        ),
                        class_="text-center"
                    ),
                    ui.div(
                        ui.div("📍", class_="text-4xl font-light text-green-600 mb-2"),
                        ui.h3("Local Facilities", class_="text-xl font-medium text-gray-900 mb-2"),
                        ui.p(
                            "Find nearby recycling centers and disposal locations instantly.",
                            class_="text-gray-600"
                        ),
                        class_="text-center"
                    ),
                    class_="grid md:grid-cols-3 gap-6"
                ),
                class_="bg-white rounded-3xl shadow-sm border border-gray-100 p-8 mb-12"
            ),
            
            # Call to Action
            ui.div(
                ui.input_action_link(
                    "nav_to_home",
                    "Try RecycLens Now",
                    class_="bg-gradient-to-r from-green-500 to-green-600 text-white px-8 py-4 rounded-2xl text-lg font-medium hover:from-green-600 hover:to-green-700 transition-all shadow-lg hover:shadow-xl"
                ),
                class_="text-center"
            ),
            class_="max-w-7xl mx-auto px-6 py-16"
        ),
        class_="min-h-screen bg-gradient-to-b from-gray-50 to-white"
    )


app_ui = ui.page_fluid(
    ui.include_css("static/styles.css"),
    ui.tags.script("""
        // Make file upload container clickable
        function setupFileUpload() {
            const containers = document.querySelectorAll('.file-upload-container');
            containers.forEach(function(container) {
                const fileInput = container.querySelector('input[type="file"]');
                if (fileInput) {
                    // Remove existing listeners
                    const newContainer = container.cloneNode(true);
                    container.parentNode.replaceChild(newContainer, container);
                    const newFileInput = newContainer.querySelector('input[type="file"]');
                    if (newFileInput) {
                        // Make entire container clickable
                        newContainer.addEventListener('click', function(e) {
                            if (e.target !== newFileInput && e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
                                e.preventDefault();
                                e.stopPropagation();
                                newFileInput.click();
                            }
                        });
                    }
                }
            });
        }
        
        // Run on page load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupFileUpload);
        } else {
            setupFileUpload();
        }
        
        // Re-run after Shiny updates the DOM
        if (window.Shiny) {
            document.addEventListener('shiny:connected', setupFileUpload);
            document.addEventListener('shiny:value', setupFileUpload);
        }
    """),
    nav_bar(),
    ui.div(
        ui.output_ui("main_content"),
        class_="min-h-screen bg-gradient-to-b from-gray-50 to-white"
    ),
    ui.div(
        ui.div(
            ui.p(
                "© 2025 RecycLens. Making recycling simple.",
                class_="text-sm text-gray-400"
            ),
            class_="max-w-7xl mx-auto px-6 text-center"
        ),
        class_="mt-32 py-12 border-t border-gray-100"
    ),
    title="RecycLens - Know What to Recycle, Instantly"
)


def server(input, output, session):
    """Server logic for the Shiny app."""
    # Reactive values
    current_page = reactive.Value("home")
    analysis_stage = reactive.Value(STAGE_IDLE)
    vision_result = reactive.Value(None)
    analysis_result = reactive.Value(None)
    error_message = reactive.Value(None)
    image_preview_data = reactive.Value(None)
    show_results = reactive.Value(False)
    
    @reactive.effect
    @reactive.event(input.nav_to_how_it_works)
    def _():
        """Handle navigation to How it Works page."""
        current_page.set("how-it-works")
    
    @reactive.effect
    @reactive.event(input.nav_to_home)
    def _():
        """Handle navigation back to home page."""
        current_page.set("home")
    
    @reactive.effect
    @reactive.event(input.nav_logo)
    def _():
        """Handle logo click to return home."""
        current_page.set("home")
    
    @output
    @render.ui
    def main_content():
        """Render main content based on current page."""
        # Simple navigation - use reactive value
        if current_page.get() == "how-it-works":
            return how_it_works_page()
        else:
            return ui.div(
                hero_section(),
                ui.div(
                    ui.div(
                        input_form(),
                        ui.output_ui("results_panel"),
                        class_="flex items-start justify-center gap-8" if show_results.get() else ""
                    ),
                    class_="max-w-7xl mx-auto px-6"
                ),
                facility_map_section(),
                class_="max-w-7xl mx-auto px-6"
            )
    
    @output
    @render.image
    def image_preview():
        """Display image preview."""
        file_info = input.image()
        if file_info is None or len(file_info) == 0:
            return None
        
        file = file_info[0]
        image_preview_data.set(file)
        
        # Return image data with fixed size and center alignment
        return {"src": file["datapath"], "width": "400px", "class": "rounded-2xl max-w-full block mx-auto"}
    
    @output
    @render.ui
    def remove_image_ui():
        """Remove image button."""
        if input.image() is None or len(input.image()) == 0:
            return None
        
        return ui.div(
            ui.tags.button(
                "Remove",
                class_="mt-2 text-sm text-red-600 hover:text-red-700 px-4 py-2 rounded-full border border-red-300 bg-white hover:bg-red-50 transition-colors",
                onclick="document.getElementById('image').value = ''; location.reload();"
            ),
            class_="flex justify-center"
        )
    
    @output
    @render.ui
    def error_display():
        """Display error messages."""
        error = error_message.get()
        if error is None:
            return None
        
        return ui.div(
            ui.p(error, class_="text-sm text-red-600"),
            class_="mb-6 p-4 bg-red-50 border border-red-200 rounded-2xl"
        )
    
    @output
    @render.ui
    def progress_indicator():
        """Display progress indicator based on current stage."""
        stage = analysis_stage.get()
        if stage == STAGE_IDLE:
            return None
        
        stage_labels = {
            STAGE_ANALYZING_IMAGE: "Analyzing image...",
            STAGE_ANALYZING_RECYCLABILITY: "Checking recyclability...",
            STAGE_GEOCODING: "Finding facilities...",
        }
        label = stage_labels.get(stage, "Analyzing...")
        
        return ui.div(
            ui.div(
                ui.span("⏳", class_="animate-spin"),
                ui.span(label, class_="ml-2"),
                class_="flex items-center text-green-600"
            ),
            class_="mb-6"
        )
    
    @reactive.effect
    @reactive.event(input.check_button)
    async def analyze_item():
        """Handle item analysis with staged progress."""
        # Reset state
        error_message.set(None)
        show_results.set(False)
        analysis_result.set(None)
        vision_result.set(None)
        
        # Validation
        file_info = input.image()
        location = input.location()
        
        if file_info is None or len(file_info) == 0:
            error_message.set("Please upload an image")
            return
        
        if not location or not location.strip():
            error_message.set("Please enter your location")
            return
        
        try:
            # Stage 1: Analyze image
            analysis_stage.set(STAGE_ANALYZING_IMAGE)
            image_base64 = convert_image_to_base64(file_info[0])
            vision_res = await analyze_vision(image_base64)
            vision_result.set(vision_res)
            
            # Stage 2: Analyze recyclability
            analysis_stage.set(STAGE_ANALYZING_RECYCLABILITY)
            context = input.context() or ""
            analysis_res = await analyze_recyclability(vision_res, location.strip(), context)
            analysis_result.set(analysis_res)
            
            # Stage 3: Geocoding (happens in frontend)
            analysis_stage.set(STAGE_GEOCODING)
            await asyncio.sleep(0.5)  # Brief delay for UX
            
            # Complete
            analysis_stage.set(STAGE_COMPLETE)
            show_results.set(True)
        except Exception as e:
            analysis_stage.set(STAGE_ERROR)
            error_message.set(f"Analysis failed: {str(e)}")
            print(f"Analysis error: {e}")
    
    @output
    @render.ui
    def results_panel():
        """Display analysis results."""
        result = analysis_result.get()
        if result is None or not show_results.get():
            return None
        
        is_recyclable = result.get("isRecyclable", False)
        category = result.get("category", "")
        bin_type = result.get("bin", "")
        material_desc = result.get("materialDescription", "")
        instructions = result.get("instructions", [])
        confidence = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")
        
        icon = "✅" if is_recyclable else "❌"
        color_class = "text-green-600" if is_recyclable else "text-red-600"
        bg_class = "bg-green-100" if is_recyclable else "bg-red-100"
        
        instruction_items = [
            ui.tags.li(inst, class_="text-sm text-gray-700 leading-relaxed")
            for inst in instructions
        ] if instructions else []
        
        return ui.div(
            ui.div(
                ui.div(
                    ui.span(icon, class_="text-4xl"),
                    class_="w-20 h-20 mx-auto mb-4 rounded-full flex items-center justify-center " + bg_class
                ),
                ui.h3(
                    "Recyclable" if is_recyclable else "Not Recyclable",
                    class_="text-3xl font-light mb-2 " + color_class
                ),
                ui.p(
                    reasoning or ("This item can be recycled" if is_recyclable else "This item cannot be recycled"),
                    class_="text-gray-600 text-sm leading-relaxed mb-4"
                ),
                class_="text-center mb-6"
            ),
            # Category
            ui.div(
                ui.p("CATEGORY", class_="text-xs font-medium text-gray-500 mb-3"),
                ui.div(
                    ui.span(category, class_="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"),
                    ui.span(bin_type, class_="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm"),
                    ui.span(material_desc, class_="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm") if material_desc else None,
                    class_="flex flex-wrap gap-2"
                ),
                class_="mb-6"
            ),
            # Instructions
            ui.div(
                ui.p("INSTRUCTIONS", class_="text-xs font-medium text-gray-500 mb-3"),
                ui.tags.ol(*instruction_items, class_="list-decimal list-inside space-y-2"),
                class_="mb-6 pt-6 border-t border-gray-100"
            ) if instruction_items else None,
            # Confidence
            ui.div(
                ui.p("CONFIDENCE", class_="text-xs font-medium text-gray-500 mb-2"),
                ui.div(
                    ui.div(
                        ui.div(
                            class_="bg-green-500 h-2 rounded-full transition-all",
                            style=f"width: {confidence * 100}%"
                        ),
                        class_="flex-1 bg-gray-200 rounded-full h-2"
                    ),
                    ui.span(
                        f"{int(confidence * 100)}%",
                        class_="text-xs text-gray-600 ml-2"
                    ),
                    class_="flex items-center gap-2"
                ),
                class_="pt-6 border-t border-gray-100"
            ),
            class_="bg-white rounded-3xl shadow-sm border border-gray-100 p-8 transition-all duration-700 ease-in-out w-[40%]"
        )
    
    @output
    @render.ui
    def facility_map():
        """Display facility map with Mapbox."""
        result = analysis_result.get()
        if result is None:
            return ui.div(
                ui.p(
                    "Upload an item to find nearby facilities",
                    class_="text-gray-400"
                ),
                class_="h-96 bg-gradient-to-br from-green-50 to-blue-50 rounded-2xl flex items-center justify-center"
            )
        
        facilities = result.get("facilities", [])
        if not facilities:
            return ui.div(
                ui.p(
                    "No facilities found for this location",
                    class_="text-gray-400"
                ),
                class_="h-96 bg-gradient-to-br from-green-50 to-blue-50 rounded-2xl flex items-center justify-center"
            )
        
        # Generate Mapbox map HTML
        map_html = generate_mapbox_html(facilities, MAPBOX_TOKEN)
        return ui.HTML(map_html)
    
    @output
    @render.ui
    def facility_cards():
        """Display facility cards."""
        result = analysis_result.get()
        if result is None:
            return None
        
        facilities = result.get("facilities", [])
        if not facilities:
            return ui.div(
                ui.p(
                    "No facilities found for this location",
                    class_="text-gray-400"
                ),
                class_="col-span-3 text-center py-8"
            )
        
        cards = [
            ui.div(
                ui.div(
                    ui.div(
                        ui.h4(facility.get("name", ""), class_="font-medium text-gray-900 mb-1"),
                        ui.div(
                            ui.span("📍", class_="w-4 h-4"),
                            ui.span(facility.get("address", ""), class_="text-sm text-gray-600"),
                            class_="flex items-center gap-1 text-sm mb-2"
                        ),
                        class_="flex-1"
                    ),
                    ui.a(
                        "🔗",
                        href=facility.get("url", "#"),
                        target="_blank",
                        class_="text-green-600 hover:text-green-700 transition-colors"
                    ) if facility.get("url") and facility.get("url") != "#" else None,
                    class_="flex items-start justify-between mb-3"
                ),
                ui.div(
                    ui.span(
                        facility.get("type", ""),
                        class_="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium"
                    ),
                    class_="flex items-center gap-2 mb-3"
                ),
                ui.p(
                    facility.get("notes", ""),
                    class_="text-sm text-gray-600 leading-relaxed"
                ) if facility.get("notes") else None,
                class_="bg-gray-50 rounded-2xl p-6 hover:bg-gray-100 transition-colors"
            )
            for facility in facilities
        ]
        
        return ui.div(
            *cards,
            class_="p-8 grid grid-cols-1 md:grid-cols-3 gap-4"
        )


def generate_mapbox_html(facilities: list, token: str) -> str:
    """Generate HTML/JavaScript for Mapbox map."""
    if not token:
        return '<div class="h-96 bg-gradient-to-br from-green-50 to-blue-50 rounded-2xl flex items-center justify-center"><p class="text-gray-400">Mapbox token not configured</p></div>'
    
    # Convert facilities to JavaScript array
    facilities_js = []
    for facility in facilities:
        facilities_js.append({
            "name": facility.get("name", ""),
            "address": facility.get("address", ""),
            "type": facility.get("type", ""),
            "url": facility.get("url", ""),
            "coordinates": facility.get("coordinates", None)
        })
    
    facilities_json = str(facilities_js).replace("'", '"')
    
    return f'''
    <div id="map" class="h-96 rounded-2xl overflow-hidden relative"></div>
    <script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
    <link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet" />
    <script>
        mapboxgl.accessToken = '{token}';
        const facilities = {facilities_json};
        const map = new mapboxgl.Map({{
            container: 'map',
            style: 'mapbox://styles/mapbox/light-v11',
            center: [-98.5795, 39.8283],
            zoom: 3
        }});
        
        // Geocode facilities and add markers
        async function geocodeAndAddMarkers() {{
            const geocodePromises = facilities.map(async (facility) => {{
                if (facility.coordinates && facility.coordinates.length === 2) {{
                    return {{
                        ...facility,
                        lng: facility.coordinates[0],
                        lat: facility.coordinates[1]
                    }};
                }}
                try {{
                    const response = await fetch(
                        `https://api.mapbox.com/geocoding/v5/mapbox.places/${{encodeURIComponent(facility.address)}}.json?access_token=${{mapboxgl.accessToken}}`
                    );
                    const data = await response.json();
                    if (data.features && data.features.length > 0) {{
                        const [lng, lat] = data.features[0].center;
                        return {{
                            ...facility,
                            lng: lng,
                            lat: lat
                        }};
                    }}
                }} catch (error) {{
                    console.error('Geocoding error:', error);
                }}
                return null;
            }});
            
            const facilitiesWithCoords = (await Promise.all(geocodePromises)).filter(f => f !== null);
            if (facilitiesWithCoords.length === 0) return;
            
            // Calculate bounds
            const lngs = facilitiesWithCoords.map(f => f.lng);
            const lats = facilitiesWithCoords.map(f => f.lat);
            const bounds = new mapboxgl.LngLatBounds();
            facilitiesWithCoords.forEach(f => bounds.extend([f.lng, f.lat]));
            map.fitBounds(bounds, {{ padding: 50 }});
            
            // Add markers
            facilitiesWithCoords.forEach((facility, index) => {{
                const el = document.createElement('div');
                el.className = 'marker';
                el.style.width = '32px';
                el.style.height = '32px';
                el.style.borderRadius = '50%';
                el.style.backgroundColor = '#22c55e';
                el.style.border = '4px solid white';
                el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                el.style.cursor = 'pointer';
                
                const popup = new mapboxgl.Popup({{ anchor: 'bottom' }})
                    .setHTML(`
                        <div style="padding: 8px; min-width: 200px;">
                            <h3 style="font-weight: 500; margin-bottom: 4px;">${{facility.name}}</h3>
                            <p style="font-size: 12px; color: #666; margin-bottom: 8px;">${{facility.address}}</p>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="padding: 2px 8px; background: #dcfce7; color: #15803d; border-radius: 9999px; font-size: 11px;">
                                    ${{facility.type}}
                                </span>
                                ${{facility.url && facility.url !== '#' ? `<a href="${{facility.url}}" target="_blank" style="color: #22c55e; font-size: 11px;">Directions</a>` : ''}}
                            </div>
                        </div>
                    `);
                
                new mapboxgl.Marker(el)
                    .setLngLat([facility.lng, facility.lat])
                    .setPopup(popup)
                    .addTo(map);
            }});
        }}
        
        map.on('load', geocodeAndAddMarkers);
    </script>
    '''


app = App(app_ui, server)

