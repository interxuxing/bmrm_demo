# -*- coding: utf-8 -*-
"""
__author__ = 'LIMU'
"""
import hashlib

# import flickrapi
try:
    import xml.etree.cElementTree as etree
except ImportError:
    import xml.etree.ElementTree as etree


API_KEY = '1c9a5bdda3c868dbcc1aa36cc32d0296'
API_SECRET = '1ae2da39604d942d'

def sign(dictionary):
    """Calculate the flickr signature for a set of params.

    data

        a hash of all the params and values to be hashed, e.g.

        “{"api_key":"AAAA", "auth_token":"TTTT", "key":

        u"value".encode(‘utf-8′)}“

    """

    data = [API_SECRET] #你的Secret值

    #dictonary就是参数的字典表示，如{"api_key":"aaa","auth_token":"bbb"}

    for key in sorted(dictionary.keys()):

        data.append(key)

        datum = dictionary[key]

        if isinstance(datum, unicode):

            raise IllegalArgumentException("No Unicode allowed, "

                    "argument %s (%r) should have been UTF-8 by now"

                    % (key, datum))

        data.append(datum)

    # data.append('methodflickr.auth.getFrob')
    # data.append('flickr.auth.getToken')
    md5_hash = hashlib.md5()

    md5_hash.update("".join(data))

    return md5_hash.hexdigest()


if __name__ == '__main__':
    # step-1, get frop value
    # method = 'flickr.auth.getFrob'
    # dictionary = {'api_key':API_KEY, 'method':method}
    # arg_sig = sign(dictionary)
    #
    # http_request = 'http://flickr.com/services/rest/?api_key=%s&method=%s&api_sig=%s' % \
    #                (API_KEY, method, arg_sig)
    # print http_request

    # step-2, get athorizate page
    # frob = '72157642592842773-fcd076017ae8d0cd-458342'
    # method = 'flickr.auth.getToken'
    # dictionary = {'api_key':API_KEY, 'frob':frob, 'perms':'write'}
    # api_sig = sign(dictionary)
    #
    # http_request = 'http://flickr.com/services/auth/?api_key=%s&perms=write&frob=%s&api_sig=%s' % \
    #                (API_KEY, frob, api_sig)
    # print http_request


    # step-3, use getToken method to get token
    frob = '72157642593965443-44a1012e976c9f22-458342'
    method = 'flickr.auth.getToken'
    dictionary = {'api_key':API_KEY, 'frob':frob, 'method':method}
    api_sig = sign(dictionary)
    http_request = 'http://flickr.com/services/rest/?method=%s&api_key=%s&frob=%s&api_sig=%s' % \
                   (method, API_KEY, frob, api_sig)

    print http_request

    # photo_usr_id = '72511036@N00'
    #
    # flickr = flickrapi.FlickrAPI(my_api_key, format='etree')
    # # photos = flickr.photos_search(user_id=photo_usr_id, per_page='10')
    # sets = flickr.photosets_getList(user_id=photo_usr_id)
    # flickr.photos.getAllContexts(my_api_key,'72157626655196872')
    # print sets

