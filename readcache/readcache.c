/*  readcache.c: Utility to dump LIGO cache files to ASCII columns.
    Copyright (C) Will M. Farr <will.farr@ligo.org>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */

/** USAGE: readcache --cache <cache file> --start <GPS time> --length <len in seconds> --channel <channel name> --output <output file> */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <lal/FrameCache.h>
#include <lal/FrameStream.h>

#include <lal/TimeSeries.h>
#include <lal/LALDatatypes.h>

#include <lal/Date.h>

#include <getopt.h>

#include <math.h>

typedef enum {
  OPT_CACHE = 1,
  OPT_GPS_START,
  OPT_LEN,
  OPT_CHAN,
  OPT_OUT
} option_enum;

static struct option options[] = {
  {"cache", required_argument, NULL, OPT_CACHE},
  {"start", required_argument, NULL, OPT_GPS_START},
  {"length", required_argument, NULL, OPT_LEN},
  {"channel", required_argument, NULL, OPT_CHAN},
  {"output", required_argument, NULL, OPT_OUT},
  {NULL, 0, NULL, 0}
};

int main(int argc, char *argv[]) {
  int opt;
  char *cache_name = NULL;
  char *channel = NULL;
  LIGOTimeGPS start, gpst;
  double t = -1.0;
  double length = 0.0;
  REAL8TimeSeries *data = NULL;
  FILE *output = NULL;
  FrCache *cache = NULL;
  FrStream *stream = NULL;
  size_t i;
  LALStatus status;

  memset(&start, 0, sizeof(start));

  while ((opt = getopt_long(argc, argv, "", options, NULL)) != -1) {
    switch (opt) {
    case OPT_CACHE:
      cache_name = optarg;
      break;

    case OPT_GPS_START:
      XLALStrToGPS(&start, optarg, NULL);
      t = XLALGPSGetREAL8(&start);
      break;

    case OPT_LEN:
      length = atof(optarg);
      break;

    case OPT_CHAN:
      channel = optarg;
      break;

    case OPT_OUT:
      output = fopen(optarg, "w");
      break;

    default:
      fprintf(stderr, "Unrecognized option!\n");
      exit(1);
      break;
    }
  }

  /* Argument Checks */
  if (length == 0.0) {
    fprintf(stderr, "Bad length argument.\n");
    exit(1);
  }
  if (cache_name == NULL) {
    fprintf(stderr, "Must specify a cache name.\n");
    exit(1);
  }
  if (t == -1.0) {
    fprintf(stderr, "Must specify a start time.\n");
    exit(1);
  }
  if (channel == NULL) {
    fprintf(stderr, "Must specify a channel.\n");
    exit(1);
  }
  if (output == NULL) {
    fprintf(stderr, "Could not open output file.\n");
    exit(1);
  }

  cache = XLALFrImportCache(cache_name);
  if (cache == NULL) {
    fprintf(stderr, "Error reading cache file.\n");
    exit(1);
  }
  stream = XLALFrCacheOpen(cache);
  if (stream == NULL) {
    fprintf(stderr, "Error opening cache.\n");
    exit(1);
  }
  data = XLALFrInputREAL8TimeSeries(stream, channel, &start, length, 0);
  if (data == NULL) {
    fprintf(stderr, "Error converting cache to time series, starting at %g, for %g seconds.\n", XLALGPSGetREAL8(&start), length);
    fprintf(stderr, "Channel = %s\n", channel);
    exit(1);
  }

  for (i = 0; i < data->data->length; i++) {
    char *tstring = NULL;
    gpst = data->epoch; /* copy of epoch */
    XLALGPSAdd(&gpst, data->deltaT*i);
    tstring = XLALGPSToStr(NULL, &gpst);
    fprintf(output, "%s %g\n", tstring, data->data->data[i]);
    XLALFree(tstring);
  }

  LALFrClose(&status, &stream);
  LALDestroyFrCache(&status, &cache);
  XLALDestroyREAL8TimeSeries(data);
  fclose(output);
}
