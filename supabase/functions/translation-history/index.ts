import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import { corsHeaders } from '../_shared/cors.ts';

/*
  Before using this Edge Function, create the translations table in your Supabase project:
  
  CREATE TABLE translations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sourceText TEXT NOT NULL,
    translatedText TEXT NOT NULL,
    fromLang TEXT NOT NULL,
    toLang TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now(),
    isFavorite BOOLEAN DEFAULT false
  );
*/

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_ANON_KEY')!,
      { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
    );

    const { method } = req;
    let body;
    if (req.body) {
      body = await req.json();
    }

    if (method === 'GET') {
      const { data: history, error } = await supabaseClient
        .from('translations')
        .select('*')
        .order('timestamp', { ascending: false });

      if (error) throw error;

      return new Response(JSON.stringify({ history }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      });
    } else if (method === 'POST') {
      const { item } = body;
      const { data, error } = await supabaseClient
        .from('translations')
        .insert([
          {
            sourceText: item.sourceText,
            translatedText: item.translatedText,
            fromLang: item.fromLang,
            toLang: item.toLang,
            isFavorite: item.isFavorite,
          },
        ])
        .select();

      if (error) throw error;

      return new Response(JSON.stringify({ success: true, data }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 201,
      });
    } else if (method === 'PUT') {
      const { id } = body;
      const { data: current, error: fetchError } = await supabaseClient
        .from('translations')
        .select('isFavorite')
        .eq('id', id)
        .single();

      if (fetchError) throw fetchError;

      const { error: updateError } = await supabaseClient
        .from('translations')
        .update({ isFavorite: !current.isFavorite })
        .eq('id', id);

      if (updateError) throw updateError;

      return new Response(JSON.stringify({ success: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      });
    } else if (method === 'DELETE') {
      const { id } = body;
      const { error } = await supabaseClient
        .from('translations')
        .delete()
        .eq('id', id);

      if (error) throw error;

      return new Response(JSON.stringify({ success: true }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      });
    }

    return new Response(JSON.stringify({ error: 'Method Not Allowed' }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 405,
    });
  } catch (err) {
    return new Response(String(err?.message ?? err), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    });
  }
});