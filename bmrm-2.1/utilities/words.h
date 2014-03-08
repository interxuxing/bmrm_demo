/* Copyright (c) 2010, NICTA 
 * All rights reserved. 
 * 
 * The contents of this file are subject to the Mozilla Public License 
 * Version 1.1 (the "License"); you may not use this file except in 
 * compliance with the License. You may obtain a copy of the License at 
 * http://www.mozilla.org/MPL/ 
 * 
 * Software distributed under the License is distributed on an "AS IS" 
 * basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the 
 * License for the specific language governing rights and limitations 
 * under the License. 
 * 
 * Author: James Petterson (james.petterson@nicta.com.au)
 */

#ifndef _WORDS_H
#define _WORDS_H

#include <ctype.h>

#define MAX_WORDS (40<<10)

/**
 * words: a class to manipulate words in a text
 */
class words
{
	char  *text;
	int    nwords;
	size_t tot_length;
	char  *w_begin[MAX_WORDS];
	char  *w_end  [MAX_WORDS];

public:
	/**
	 * \brief Constructor: parses text, creating pointers for the begin and end of every word.
	 */
	words(char *_text, bool remove_punct=true) : text(_text)
	{
		char *p;
		int i, n;

		if (remove_punct) /* code below duplicated for performance reasons */
		{
			/* find the beginning and end of every word on text */
			for (i=n=0, p=text; (i<MAX_WORDS) && (*p); i++)
			{
				/* find and mark the beginning */
				while ( (*p) && (isspace(*p)||ispunct(*p)||(!isprint(*p))) ) p++;
				if (!*p) break;
				w_begin[i] = p;
				n++;

				/* find and mark the end */
				while ( (*p) && !(isspace(*p)||ispunct(*p)||(!isprint(*p))) ) p++;
				w_end[i] = p;
				if (!*p) break;
			}
		}
		else
		{
			/* find the beginning and end of every word on text */
			for (i=n=0, p=text; (i<MAX_WORDS) && (*p); i++)
			{
				/* find and mark the beginning */
				while ( (*p) && isspace(*p) ) p++;
				if (!*p) break;
				w_begin[i] = p;
				n++;

				/* find and mark the end */
				while ( (*p) && !isspace(*p) ) p++;
				w_end[i] = p;
				if (!*p) break;
			}
		}

		if (i==MAX_WORDS)
		{
			std::cerr <<  "More than " << MAX_WORDS << " words in text!" << std::endl;
		}

		nwords = n;
		tot_length = p-text;
	}

	/**
	 * \brief Returns the number of words in text.
	 */
	int count() 
	{ 
		return nwords; 
	}

	/**
	 * \brief Returns pointer to begin of n-th word in text.
	 */
	char *begin(int n)
	{ 
		return w_begin[n];
	}

	/**
	 * \brief Returns pointer to end of n-th word in text.
	 */
	char *end(int n)
	{ 
		return w_end[n];
	}

	/**
	 * \brief Returns length of n-th word in text.
	 */
	size_t len(int n)
	{
		return (w_end[n]-w_begin[n]);
	}

	/**
	 * \brief Returns total length of text.
	 */
	size_t total_length()
	{
		return tot_length;
	}

};

#endif /* _WORDS_H */

