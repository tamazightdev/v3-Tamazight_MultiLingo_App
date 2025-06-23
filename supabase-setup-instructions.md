# 🚀 Complete Supabase Setup Guide for Online Mode

## 📋 Prerequisites
- ✅ Supabase project created (tamazightdev's Project)
- ✅ Supabase CLI installed: `npm install -g supabase`

## 🗄️ Step 1: Create Database Table

### Where to Run the SQL
1. **Go to your Supabase Dashboard**: https://supabase.com/dashboard
2. **Select your project**: "tamazightdev's Project"
3. **Navigate to SQL Editor**: Click "SQL Editor" in the left sidebar
4. **Create a new query**: Click "New Query" button

### SQL Code to Run
Copy and paste this exact SQL code into the SQL Editor:

```sql
-- Create translations table for storing translation history
CREATE TABLE IF NOT EXISTS translations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  sourceText TEXT NOT NULL,
  translatedText TEXT NOT NULL,
  fromLang TEXT NOT NULL,
  toLang TEXT NOT NULL,
  timestamp TIMESTAMPTZ DEFAULT now(),
  isFavorite BOOLEAN DEFAULT false
);

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_translations_timestamp ON translations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_translations_favorite ON translations(isFavorite) WHERE isFavorite = true;

-- Enable Row Level Security (RLS) for data protection
ALTER TABLE translations ENABLE ROW LEVEL SECURITY;

-- Create policy to allow all operations for now (you can restrict this later)
CREATE POLICY "Allow all operations on translations" ON translations
  FOR ALL USING (true);

-- Verify table creation
SELECT 'Table created successfully!' as status;
```

### Execute the SQL
1. **Paste the SQL code** into the editor
2. **Click "Run"** button (or press Ctrl+Enter)
3. **Verify success**: You should see "Table created successfully!" in the results

## 🔧 Step 2: Deploy Edge Function

### Initialize Supabase in Your Project
Open terminal in your project root and run:

```bash
# Login to Supabase CLI
supabase login

# Link your project (replace with your actual project ID)
supabase link --project-ref YOUR_PROJECT_REF

# Initialize functions (if not already done)
supabase functions new translation-history
```

### Deploy the Edge Function
The edge function code is already created in your project. Deploy it:

```bash
# Deploy the translation-history function
supabase functions deploy translation-history

# Verify deployment
supabase functions list
```

## 🔑 Step 3: Get Your Credentials

### Find Your Project Credentials
1. **Go to Project Settings**: Click the gear icon ⚙️ in your dashboard
2. **Navigate to API**: Click "API" in the settings menu
3. **Copy the following values**:

```
Project URL: https://your-project-id.supabase.co
Anon Key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Update Environment Variables
Create or update your `.env` file in the project root:

```env
# Supabase Configuration for Online Mode
EXPO_PUBLIC_SUPABASE_URL="https://your-project-id.supabase.co"
EXPO_PUBLIC_SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Optional: Gemma-3 API Configuration
EXPO_PUBLIC_GEMMA_API_KEY="your-gemma-api-key"
```

## 🧪 Step 4: Test the Setup

### Test Database Connection
1. **Start your app**: `pnpm dev`
2. **Go to Settings**: Switch to "Online" mode
3. **Make a translation**: Go to translate tab and translate some text
4. **Check history**: Go to history tab and verify it shows "Cloud Storage"

### Verify in Supabase Dashboard
1. **Go to Table Editor**: Click "Table Editor" in Supabase dashboard
2. **Select translations table**: You should see your test translations
3. **Check data**: Verify the translations are stored correctly

## 🔍 Step 5: Troubleshooting

### Common Issues and Solutions

#### Issue: "Supabase not configured" error
**Solution**: 
- Verify your `.env` file has the correct credentials
- Restart your development server after updating `.env`
- Check that environment variable names match exactly

#### Issue: Edge function not found
**Solution**:
```bash
# Redeploy the function
supabase functions deploy translation-history --no-verify-jwt

# Check function logs
supabase functions logs translation-history
```

#### Issue: Database connection failed
**Solution**:
- Verify your project URL is correct
- Check that the anon key is valid
- Ensure your project is not paused

#### Issue: CORS errors
**Solution**: The CORS configuration is already included in the edge function. If you still see CORS errors:
- Redeploy the edge function
- Check browser console for specific error details

### Debug Commands
```bash
# Check Supabase CLI status
supabase status

# View function logs
supabase functions logs translation-history --follow

# Test edge function locally
supabase functions serve translation-history
```

## 📊 Step 6: Verify Everything Works

### Complete Test Checklist
- [ ] SQL table created successfully
- [ ] Edge function deployed without errors
- [ ] Environment variables configured
- [ ] App starts without errors
- [ ] Can switch to online mode
- [ ] Translations save to cloud storage
- [ ] History shows "Cloud Storage" indicator
- [ ] Can toggle favorites in online mode
- [ ] Can delete translations in online mode

### Expected Behavior
When everything is working correctly:

1. **Settings Screen**: Toggle shows online/offline modes
2. **Translate Screen**: Shows "Online Mode" indicator when active
3. **History Screen**: Shows "Cloud Storage • X translations"
4. **Database**: Translations appear in Supabase table editor
5. **Performance**: Smooth operation with loading indicators

## 🎯 Next Steps

### Optional Enhancements
1. **User Authentication**: Add Supabase Auth for user-specific data
2. **Real-time Updates**: Use Supabase realtime for live sync
3. **Data Migration**: Add sync between offline and online modes
4. **Advanced Queries**: Add search and filtering in the database

### Production Considerations
1. **Row Level Security**: Implement proper RLS policies
2. **Rate Limiting**: Add rate limiting to edge functions
3. **Error Monitoring**: Set up error tracking
4. **Performance Monitoring**: Monitor edge function performance

## 📞 Support

If you encounter any issues:

1. **Check the browser console** for error messages
2. **Review Supabase function logs** for backend errors
3. **Verify environment variables** are correctly set
4. **Test with a fresh browser session** to clear cache

### Useful Supabase Resources
- [Supabase Documentation](https://supabase.com/docs)
- [Edge Functions Guide](https://supabase.com/docs/guides/functions)
- [Database Guide](https://supabase.com/docs/guides/database)

---

**Status**: 🎯 Ready for Implementation
**Estimated Setup Time**: 10-15 minutes
**Difficulty**: Beginner-friendly with step-by-step instructions