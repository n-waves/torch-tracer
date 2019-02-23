#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <optional>
#include <regex>
#include <sqlite3.h>
#include <climits>

#include "nlohmann/json.hpp"

using namespace std;
using namespace nlohmann;

typedef unsigned long long int ull;

class AggregateFrame {
public:
  AggregateFrame(const string &_identifier) : identifier(_identifier), self_time(0.0), seq(nullopt) {}

  string identifier;
  double self_time;
  map<string, AggregateFrame*> children;
  optional<ull> seq;
  string function;
};

struct Row {
  ull ts;
  string filename;
  string function;
  int line;
  string tpe;
};

template <class Container>
void split(const std::string& str, Container& cont, char delim=' ')
{
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim)) {
        cont.push_back(token);
    }
}


regex stashed_seq_re("^ stashed seq=([0-9]+)$", regex_constants::extended);
regex seq_re("^ seq=([0-9]+)$", regex_constants::extended);

optional<ull> get_int(const string &str, const regex& re) {
  smatch match;
  if (regex_search(str, match, re)) {
    return stoull(match[1]);
  } else {
    return {};
  }
}

#define SQLITE_CHECK(err, db) {if((err)) { cerr << "sqlite error: " << sqlite3_errmsg((db)) << endl; exit(-1); }}

Row cpu2row(sqlite3_stmt *stmt) {
  Row row;
  row.filename = (const char*)sqlite3_column_text(stmt, 0);
  row.function = (const char*)sqlite3_column_text(stmt, 1);
  row.line = sqlite3_column_int64(stmt, 2);
  string what = (const char*)sqlite3_column_text(stmt, 3);
  if (what == "call" || what == "ccall")
    row.tpe = "start";
  else
    row.tpe = "end";
  row.ts = sqlite3_column_int64(stmt, 4);
  return row;
}

Row cuda2row(sqlite3_stmt *stmt) {
  Row row;
  row.filename = "<cuda>";
  row.line = 0;
  row.ts = sqlite3_column_int64(stmt, 0);
  row.function = (const char*)sqlite3_column_text(stmt, 1);
  int flag = sqlite3_column_int(stmt, 2);
  row.tpe = (flag == 2 ? "start" : "end");
  return row;
}

class TableMerger {
public:
  TableMerger(const string &cpu_filename, const string &cuda_filename) {
    SQLITE_CHECK(sqlite3_open_v2(cpu_filename.data(), &cpu, SQLITE_OPEN_READONLY, nullptr), cpu);
    SQLITE_CHECK(sqlite3_open_v2(cuda_filename.data(), &cuda, SQLITE_OPEN_READONLY, nullptr), cuda);

    sqlite3_prepare_v2(cpu, "SELECT f.value as filename, fn.value as function, r.line, t.value as what, r.timestamp\
      FROM RECORDS as r INNER JOIN FILENAMES as f ON r.filename = f.id\
      INNER JOIN FUNCTIONS as fn ON r.function = fn.id\
      INNER JOIN TRACE_TYPES as t ON r.what = t.id", -1, &cpu_stmt, nullptr);
    sqlite3_prepare_v2(cuda, "SELECT markers.timestamp, names.value as function, markers.flags FROM CUPTI_ACTIVITY_KIND_MARKER as markers INNER JOIN StringTable as names on markers.name = names._id_ WHERE markers.flags = 2 OR markers.flags = 4", -1, &cuda_stmt, nullptr);

    cpu_row = get_next_cpu();
    cuda_row = get_next_cuda();
  }

  optional<Row> get_next() {
    if (!cpu_row.has_value() && !cuda_row.has_value())
      return {};
    ull cpu_ts = cpu_row.has_value() ? cpu_row->ts : ULLONG_MAX;
    ull cuda_ts = cuda_row.has_value() ? cuda_row->ts : ULLONG_MAX;

    Row r;
    if (cpu_ts <= cuda_ts) {
      r = *cpu_row;
      cpu_row = get_next_cpu();
    } else {
      r = *cuda_row;
      cuda_row = get_next_cuda();
    }
    return r;
  }

  optional<Row> get_next_cpu() {
    return get_next_row(cpu_stmt, cpu2row);
  }

  optional<Row> get_next_row(sqlite3_stmt *stmt, function<Row(sqlite3_stmt*)> convert) {
    int r = sqlite3_step(stmt);
    Row row;
    switch(r) {
      case SQLITE_ROW:
        return convert(stmt);
      case SQLITE_DONE:
        return {};
      default:
        cerr << "Unexpected return value from sqlite3_step: " << r << endl;
        exit(-1);
        return {};
    }
  }

  optional<Row> get_next_cuda() {
    return get_next_row(cuda_stmt, cuda2row);
  }

  sqlite3 *cpu, *cuda;
  sqlite3_stmt *cpu_stmt, *cuda_stmt;
  optional<Row> cpu_row, cuda_row;
};

bool endsWith(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}


AggregateFrame *aggregate(TableMerger *merger) {
  auto root = new AggregateFrame("<root>\0file.py\0000"s);
  vector<AggregateFrame*> stack;
  stack.push_back(root);
  map<ull, AggregateFrame*> seq_register;
  map<string, int> op_counter;

  optional<Row> row = merger->get_next();
  if (!row)
    return root;
  ull last_ts = row->ts;
  do {
    double ts = (row->ts - last_ts) * 1E-9;
    last_ts = row->ts;

    if (endsWith(row->filename, "torchtracer.py"))
      continue;
    optional<ull> stashed = nullopt;
    optional<ull> seq = nullopt;
    if (stack.empty()) {
      cout << "stack is empty" << endl;
      stack.push_back(root);
    }
    auto function = row->function;
    if (row->filename == "<cuda>") {
      vector<string> parts;
      split(row->function, parts, ',');
      function = parts[0];
      if (parts.size() > 1) {
        stashed = get_int(parts[1], stashed_seq_re);
        seq = get_int(parts[1], seq_re);
      }
    }
    auto name = function + "\0"s + row->filename + "\0"s + to_string(row->line);
    
    auto parent = stack.back();
    parent->self_time += ts;
    if (row->tpe == "start") {
      if (stashed) {
        auto it = seq_register.find(*stashed);
        if (it != seq_register.end()) {
          auto sr = it->second;
          name = function + " (" + sr->function + ':' + to_string(*sr->seq) + ")\0"s + row->filename + "\0"s + to_string(row->line);
        } else {
          // cout << "Stashed operation not found" << endl;
        }
      }
      auto it = parent->children.find(name);
      AggregateFrame *frame;
      if (it != parent->children.end()) {
        frame = it->second;
      } else {
        frame = new AggregateFrame(name);
        parent->children[name] = frame;
        if (seq) {
          frame->seq = op_counter[function];
          op_counter[function] ++;
        }
      }
      stack.push_back(frame);
      if (seq) {
        seq_register[*seq] = frame;
        frame->function = function;
      }
    } else {
      stack.pop_back();
    }
  } while(row = merger->get_next());
  return root;
}

json frame2json(const AggregateFrame *frame) {
  json j;
  j["identifier"] = frame->identifier;
  j["self_time"] = frame->self_time;
  j["function"] = frame->function;
  if (frame->seq.has_value())
    j["seq"] = *frame->seq;
  else
    j["seq"] = nullptr;
  json ch = json::object();
  for(auto it = frame->children.begin(); it != frame->children.end(); it++)
    ch[it->first] = frame2json(it->second);
  j["children"] = ch;
  return j;
}

int main(int argc, const char **argv) {
  if (argc != 4) {
    cout << "Usage:" << endl;
    cout << "\t" << argv[0] << " [CPU-DATABASE] [CUDA-DATABASE] [OUTPUT]" << endl;
  }
  auto merger = TableMerger(argv[1], argv[2]);
  auto root = aggregate(&merger);
  auto j = frame2json(root);
  ofstream out(argv[3]);
  out << j;
  out.close();
  return 0;
}
