# Multi-Lingo AI: AI Powered On-Device Translation

## Bridging languages, preserving culture, empowering communication.

**An official entry into the Google DeepMind & Kaggle "Gemma 3n Impact Challenge."**

**Multi-Lingo AI: Tamazight (Tama-Zir) Edition**

A beautiful, production-ready mobile translation app built with Expo and React Native. This project leverages the groundbreaking, on-device capabilities of Google's **Gemma 3n** to provide private, offline-first, and culturally-aware translation. This version of the app provides seamless translation between Tamazight, Arabic, French, and English, with specialized features for emergency situations and government/parliamentary use, directly addressing the core themes of the Gemma 3n Impact Challenge.

---

## 🏆 The Gemma 3n Impact Challenge: Our Mission

This project is being submitted to the **Google DeepMind - The Gemma 3n Impact Challenge**. Our mission is to demonstrate how next-generation, on-device AI can create tangible, positive change. We believe Multi-Lingo AI directly aligns with the challenge's core goals by:

*   **Enhancing Accessibility & Aiding in Crisis Response**: By providing instant, offline translation for emergency services, medical situations, and daily communication, we break down critical barriers for speakers of Tamazight and other languages, especially in low-connectivity regions.
*   **Preserving Culture & Revolutionizing Education**: The app serves as a tool for linguistic preservation and learning, making the Tamazight language and its Tifinagh script more accessible to a global audience.
*   **Leveraging Unique Gemma 3n Capabilities**: Our app is built from the ground up to harness the privacy-first, offline-ready, and multimodal power of Gemma 3n, showcasing its efficiency and real-world utility.

### How We Meet the Judging Criteria

*   **Impact & Vision (40%)**: We address the significant real-world challenge of language barriers, which can impede access to emergency services, legal rights, and cultural participation. Our vision is a world where technology empowers linguistic diversity rather than diminishing it.
*   **Video Pitch & Storytelling (30%)**: Our video demo will showcase real-world scenarios: a tourist navigating a Moroccan souk, a government official using parliamentary terms, and a user accessing critical medical phrases during an emergency—all offline, powered by Gemma 3n.
*   **Technical Depth & Execution (30%)**: Our public code repository and this document verify our work. We have successfully fine-tuned both **Gemma-3n-E2B-IT** and **Gemma-3n-E4B-IT** on our custom datasets (see notebook: `v3-33-gemma-3n-e2b-instruct-071225-kaggle.ipynb`) to specialize them for the nuances of Tamazight translation and specific terminologies. The app's architecture allows for dynamic switching between the efficient E2B model and the high-quality E4B model, demonstrating the "mix’n’match" flexibility of Gemma 3n.

---

## 🌟 Features

### Core Translation Features
- **Offline AI Translation**: Powered by fine-tuned **Gemma 3n E2B & E4B Instruct Models** for fast, private, on-device offline translations.
- **Multi-Language Support**: Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ), Arabic, French, and English.
- **Tifinagh Keyboard**: Built-in virtual keyboard for typing in Tifinagh script.
- **Voice Input & Output**: Speech-to-text input and text-to-speech output.
- **Real-time Translation**: Instant translation as you type.

### Future Features
- **Camera Translation**: Leveraging Gemma 3n's native multimodal capabilities to translate text from images and video streams.

### Specialized Features
- **Emergency Phrases**: Pre-loaded critical phrases for medical and emergency situations.
- **Government & Parliamentary Terms**: Official terminology for legal and administrative contexts.
- **Translation History**: Save and manage your translation history with favorites.
- **Offline Functionality**: Works completely offline once installed, ensuring privacy and accessibility.

### User Experience
- **Beautiful Glass-morphism UI**: Modern, elegant interface with gradient backgrounds.
- **Haptic Feedback**: Tactile responses for better user interaction (mobile only).
- **Responsive Design**: Optimized for both mobile and web platforms.
- **Accessibility**: Screen reader support and high contrast ratios.

## 🚀 Installation

### Prerequisites
- Node.js 18+
- npm or pnpm package manager
- Expo CLI (optional, for additional development features)

### Using npm

```bash
# Clone the repository
git clone <repository-url>
cd tamazight-translate

# Install dependencies
npm install

# Start the development server
npm run dev
```

### Using pnpm

```bash
# Clone the repository
git clone <repository-url>
cd tamazight-translate

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

### Development Server

After running the development command, you'll see:

```
Starting project at /home/project
› Metro waiting on exp://192.168.1.100:8081
› Scan the QR code above with Expo Go (Android) or the Camera app (iOS)
› Press a │ open Android
› Press i │ open iOS simulator
› Press w │ open web
```

- **Web**: Press `w` to open in your browser
- **Mobile**: Scan the QR code with Expo Go app
- **iOS Simulator**: Press `i` (requires Xcode)
- **Android Emulator**: Press `a` (requires Android Studio)

## 📱 App Structure & User Flow

### Navigation Architecture

The app uses a tab-based navigation structure with four main sections:

#### 1. **Translate Tab** (Home)
- **Purpose**: Main translation interface
- **Features**:
  - Language selector with swap functionality
  - Text input with Tifinagh keyboard support
  - Voice input/output controls
  - Camera translation button
  - Real-time AI translation powered by Gemma 3n
  - Translation history saving

**User Flow**:
1. Select source and target languages
2. Input text via typing, voice, or camera
3. Tap translate for AI-powered translation
4. Use voice output to hear pronunciation
5. Save to favorites or history

#### 2. **History Tab**
- **Purpose**: Manage translation history and favorites
- **Features**:
  - Search through past translations
  - Filter by favorites
  - Delete unwanted entries
  - Re-translate or modify previous entries

**User Flow**:
1. Browse chronological translation history
2. Search for specific translations
3. Toggle favorites filter
4. Tap any item to hear pronunciation
5. Swipe or tap to delete entries

#### 3. **Emergency Tab**
- **Purpose**: Quick access to critical phrases (Crisis Response Feature)
- **Features**:
  - Pre-loaded emergency phrases
  - Medical, police, and basic needs categories
  - Priority-coded phrases (high/medium/low)
  - Instant voice output
  - Morocco emergency contact information

**User Flow**:
1. Select emergency category (Medical, Emergency, Basic Needs)
2. Choose target language
3. Tap any phrase for instant voice output
4. Access Morocco emergency numbers (15, 19, 177)

#### 4. **Government Tab**
- **Purpose**: Official and parliamentary terminology
- **Features**:
  - Parliamentary procedure phrases
  - Legal and administrative terms
  - Constitutional rights information
  - Educational system terminology
  - Public services phrases

**User Flow**:
1. Select category (Parliament, Legal, Administrative, etc.)
2. Choose target language
3. Browse context-specific phrases
4. Learn about linguistic rights in Morocco

### Key User Interactions

#### Language Selection
- Tap language buttons to open selection modal
- Use swap button to quickly reverse translation direction
- Visual feedback with smooth animations

#### Text Input Methods
1. **Typing**: Standard keyboard input
2. **Tifinagh Keyboard**: Toggle virtual Tifinagh keyboard
3. **Voice Input**: Tap microphone for speech-to-text
4. **Camera**: (Future) OCR text recognition from images

#### Translation Process
1. Enter text in source language
2. Tap the prominent "Translate" button
3. Gemma 3n processing indicator shows progress
4. Results appear with pronunciation option
5. Save to history automatically

#### Voice Features
- **Input**: Speech recognition in multiple languages
- **Output**: Text-to-speech with proper pronunciation
- **Language Detection**: Automatic language identification

## 🛠 Technical Architecture

### Frontend Framework
- **Expo SDK 52.0.30**: Cross-platform development framework
- **React Native**: Native mobile app development
- **Expo Router 4.0.17**: File-based routing system
- **TypeScript**: Type-safe development

### UI/UX Libraries
- **Expo Linear Gradient**: Beautiful gradient backgrounds
- **Expo Blur**: Glass-morphism effects
- **Lucide React Native**: Consistent icon system
- **Expo Google Fonts**: Inter font family

### Core Features
- **Expo Speech**: Text-to-speech functionality
- **Expo Camera**: OCR and image translation
- **Expo Haptics**: Tactile feedback (mobile only)
- **Better SQLite3**: Local data storage for history and favorites

### AI Integration
- **Gemma 3n Models**: The app integrates two fine-tuned models:
    - **Gemma-3n-E2B-IT**: A 2B effective parameter model for maximum on-device speed and efficiency, perfect for real-time typing and voice transcription.
    - **Gemma-3n-E4B-IT**: A 4B effective parameter model for higher-quality, nuanced translations, used for more complex sentences or when selected by the user.
- **Fine-Tuned for Purpose**: Models were fine-tuned on a custom dataset of Tamazight-English/French/Arabic pairs, including specialized legal and medical terminology.
- **Local & Private**: No internet is required for translations. All processing happens on the user's device, guaranteeing absolute privacy.
- **Optimized Performance**: We leverage Gemma 3n's novel architecture, including Per-Layer Embeddings (PLE), to achieve a minimal memory footprint, making high-quality AI translation viable on a wide range of mobile devices.

## 🌍 Supported Languages

### Primary Languages
- **Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ)**: Standard Tamazight with Tifinagh script
- **Central Atlas Tamazight**: Regional variant
- **Arabic (العربية)**: Modern Standard Arabic
- **French (Français)**: Standard French
- **English**: International English

### Language Features
- **Bidirectional Translation**: Seamlessly translate between any supported language pair.
- **Script Support**: Native support for Latin, Arabic, and Tifinagh scripts.
- **Voice Support**: High-quality text-to-speech for all languages, enhanced by Gemma 3n's improved multilingual capabilities.
- **Cultural Context**: Phrases and translations are adapted for the Moroccan context.

## 📋 Development Scripts

```bash
# Development
npm run dev          # Start development server
pnpm dev            # Start with pnpm

# Building
npm run build:web   # Build for web deployment
pnpm build:web      # Build with pnpm

# Code Quality
npm run lint        # Run ESLint
pnpm lint          # Lint with pnpm
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory to manage model paths:

```env
EXPO_PUBLIC_API_URL=https://your-api-url.com
EXPO_PUBLIC_GEMMA_3N_E2B_PATH=./models/gemma-3n-e2b-it
EXPO_PUBLIC_GEMMA_3N_E4B_PATH=./models/gemma-3n-e4b-it
```

### Expo Platform-Specific Features
The app automatically detects the platform and enables/disables features:

- **Web**: Full functionality except haptics and some native features.
- **iOS/Android**: Complete feature set including haptics and native APIs.
- **Responsive**: Adapts to different screen sizes and orientations.

## 🚀 Deployment

### Web Deployment
```bash
npm run build:web
# Deploy the dist folder to your hosting provider
```

### Mobile App Stores
1. **iOS App Store**: Use EAS Build for iOS deployment.
2. **Google Play Store**: Use EAS Build for Android deployment.
3. **Expo Updates**: Over-the-air updates for published apps.

### EAS Build (Recommended)
```bash
# Install EAS CLI
npm install -g @expo/eas-cli

# Configure EAS
eas build:configure

# Build for production
eas build --platform all
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google DeepMind & Kaggle**: For the Gemma 3n Impact Challenge and for putting state-of-the-art AI in the hands of developers.
- **Tamazight Language Community**: For their invaluable cultural and linguistic guidance.
- **Expo Team**: For the excellent development framework that makes cross-platform development a joy.
