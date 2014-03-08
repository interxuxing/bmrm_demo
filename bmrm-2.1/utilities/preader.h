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

#ifndef _PREADER_H
#define _PREADER_H

#include "bmrmexception.hpp"

/*
 * /brief Generic to_string template converter function
 */
template <typename T>
std::string to_string( const T& t )
{
	std::ostringstream oss;
	oss << t;
	return oss.str();
}

/**
 * preader: a class to read plain files line by line, using a static buffer to improve speed.
 */
class preader
{
	char  *buf, *pos;
	char   end_of_line;
	int    len, buf_size;
	FILE  *in;
	bool   done;

public:
	preader(const char *filename, int _buf_size=1<<20) : buf_size(_buf_size)
	{
		/* Opens input stream */
		in = fopen(filename, "rb");
		if (!in)
		      throw CBMRMException("can't open file "+to_string(filename), "preader");

		/* alloc memory for buffer */
		buf = new char[buf_size];
		pos = buf;

		/* reads first chunk */
		len = fread(buf, 1, buf_size-1, in);
		if (len<=0)
			throw CBMRMException("error reading file "+to_string(filename), "preader");

		buf[len] = 0;
		done = false;
		end_of_line = '\n';
	}

	bool is_done()
	{
		return done;
	}

	~preader()
	{
		if (in)	fclose(in);
		delete[] buf;
	}

	char *get_line() /* note: this is not reentrant ! */
	{
		while (!done)
		{
			char *p, *line;
			int n;

			/* searches for the end of the current line */
			p = strchr(pos, end_of_line);
		
			/* found: marks the end of the line and returns a pointer to the beginning of it */
			if (p)
			{
				*p = 0;
				line = pos;
				pos = p+1;
				return line;
			}

			/* not found; if we have a full buffer, and we assume there is no line bigger 
			 * than buf_size, we can say this is the last line... */
			if (p==buf)
			{
				done = true;
				return p;
			}

			/* if that's not the case, let's read more data */

			/* end of file ? */
			if (feof(in))
			{
				done = true;
				return (pos);
			}
			
			/* not end of file, let's read more */
			memmove(buf, pos, buf+len-pos);

			len -= (pos-buf);

			pos  = buf;

			n = fread(buf+len, 1, buf_size-len-1, in);

			if (n<0)
				throw CBMRMException("fread returned "+to_string(n), "preader");

			buf[len+n]=0;

			len += n;
		}
		return NULL;
	}
};

#endif /* _PREADER_H */

